#include "server.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;
static void close_sock(intptr_t s) { closesocket((SOCKET) s); }
static const intptr_t BAD = (intptr_t) INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
static void close_sock(intptr_t s) { close((int) s); }
static const intptr_t BAD = -1;
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

std::string nunchi_request::q(const std::string & key) const {
    size_t i = 0;
    while (i <= query.size()) {
        const size_t amp = query.find('&', i);
        const std::string kv = query.substr(i, amp == std::string::npos ? std::string::npos : amp - i);
        const size_t eq = kv.find('=');
        if (kv.substr(0, eq) == key) {
            std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1), out;
            for (size_t j = 0; j < v.size(); j++) {   // percent-decoding
                if (v[j] == '+') out += ' ';
                else if (v[j] == '%' && j + 2 < v.size()) { out += (char) strtol(v.substr(j + 1, 2).c_str(), nullptr, 16); j += 2; }
                else out += v[j];
            }
            return out;
        }
        if (amp == std::string::npos) break;
        i = amp + 1;
    }
    return "";
}

std::string nunchi_json(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char) c;
    }
    return o + "\"";
}

std::string nunchi_base64(const uint8_t * p, size_t n) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((n + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < n; i += 3) {
        const uint32_t v = p[i] << 16 | p[i + 1] << 8 | p[i + 2];
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += T[v & 63];
    }
    if (i < n) {
        const uint32_t v = p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0);
        o += T[v >> 18]; o += T[(v >> 12) & 63];
        o += i + 1 < n ? T[(v >> 6) & 63] : '=';
        o += '=';
    }
    return o;
}

static bool send_all(intptr_t s, const char * p, size_t n) {
    while (n > 0) {
        const int k = send((decltype(socket(0, 0, 0))) s, p, (int) std::min<size_t>(n, 1 << 20), MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k;
        n -= (size_t) k;
    }
    return true;
}

bool nunchi_server::start(int port, std::string & err) {
#ifdef _WIN32
    WSADATA w;
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) { err = "no network (WSAStartup)"; return false; }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    const auto s = socket(AF_INET, SOCK_STREAM, 0);
    if ((intptr_t) s == BAD) { err = "no socket"; return false; }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t) port);
    if (bind(s, (sockaddr *) &a, sizeof a) != 0) { err = "port " + std::to_string(port) + " is taken"; close_sock((intptr_t) s); return false; }
    if (listen(s, 64) != 0) { err = "can't listen"; close_sock((intptr_t) s); return false; }
    lsock = (intptr_t) s;
    running = true;
    acceptor = std::thread([this] {
        while (running) {
            fd_set rd;   // wait at most 200 ms, so stop() never waits on a blocked accept
            FD_ZERO(&rd);
            FD_SET((decltype(socket(0, 0, 0))) lsock, &rd);
            timeval wait{ 0, 200000 };
            if (select((int) lsock + 1, &rd, nullptr, nullptr, &wait) <= 0) continue;
            sockaddr_in ca{};
            socklen_t cl = sizeof ca;
            const auto c = accept((decltype(socket(0, 0, 0))) lsock, (sockaddr *) &ca, &cl);
            if ((intptr_t) c == BAD) { if (!running) break; std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
            int one = 1;
            setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof one);
#ifdef _WIN32
            DWORD tmo = 10000;
            setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, (const char *) &tmo, sizeof tmo);
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, (const char *) &tmo, sizeof tmo);
#else
            timeval tv{ 10, 0 };
            setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#ifdef SO_NOSIGPIPE
            setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
#endif
            std::thread([this, c] { serve((intptr_t) c); }).detach();
        }
    });
    return true;
}

void nunchi_server::stop() {
    if (!running) return;
    running = false;
    if (acceptor.joinable()) acceptor.join();
    close_sock(lsock);
    std::lock_guard<std::mutex> l(cm);
    for (auto & c : clients) { std::lock_guard<std::mutex> k(c->m); c->dead = true; c->cv.notify_all(); }
}

nunchi_server::~nunchi_server() { stop(); }

int nunchi_server::listeners() const { std::lock_guard<std::mutex> l(cm); return (int) clients.size(); }
int nunchi_server::voice_listeners() const {
    std::lock_guard<std::mutex> l(cm);
    int n = 0;
    for (auto & c : clients) n += c->audio;
    return n;
}

void nunchi_server::send(const std::string & event, const std::string & json, bool keep, bool audio) {
    const std::string msg = "event: " + event + "\ndata: " + json + "\n\n";
    std::lock_guard<std::mutex> l(cm);
    if (keep) {
        history.push_back(msg);
        while (history.size() > 80) history.pop_front();
    }
    for (auto & c : clients) {
        if (audio && !c->audio) continue;
        std::lock_guard<std::mutex> k(c->m);
        if (c->out.size() > 4000) { c->dead = true; c->cv.notify_all(); continue; }   // a phone that stopped reading
        c->out.push_back(msg);
        c->cv.notify_all();
    }
}

void nunchi_server::serve(intptr_t s) {
    std::string req;
    char buf[16384];
    size_t head_end = std::string::npos;
    while ((head_end = req.find("\r\n\r\n")) == std::string::npos && req.size() < 65536) {
        const int k = recv((decltype(socket(0, 0, 0))) s, buf, sizeof buf, 0);
        if (k <= 0) { close_sock(s); return; }
        req.append(buf, (size_t) k);
    }
    if (head_end == std::string::npos) { close_sock(s); return; }
    nunchi_request r;
    {
        const size_t sp1 = req.find(' '), sp2 = req.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) { close_sock(s); return; }
        r.method = req.substr(0, sp1);
        const std::string target = req.substr(sp1 + 1, sp2 - sp1 - 1);
        const size_t qm = target.find('?');
        r.path = target.substr(0, qm);
        r.query = qm == std::string::npos ? "" : target.substr(qm + 1);
    }
    size_t len = 0;
    {
        std::string h = req.substr(0, head_end);
        for (auto & c : h) c = (char) tolower((unsigned char) c);
        const size_t cl = h.find("\r\ncontent-length:");
        if (cl != std::string::npos) len = (size_t) strtoull(h.c_str() + cl + 17, nullptr, 10);
    }
    if (len > (16u << 20)) { close_sock(s); return; }
    r.body = req.substr(head_end + 4);
    while (r.body.size() < len) {
        const int k = recv((decltype(socket(0, 0, 0))) s, buf, sizeof buf, 0);
        if (k <= 0) { close_sock(s); return; }
        r.body.append(buf, (size_t) k);
    }
    if (r.method == "GET" && r.path == "/events") { events(s, r.q("audio") == "1"); return; }
    nunchi_reply rep;
    if (handle) rep = handle(r);
    else rep.code = 404;
    const char * status = rep.code == 200 ? "OK" : rep.code == 404 ? "Not Found" : "Error";
    const std::string head = "HTTP/1.1 " + std::to_string(rep.code) + " " + status + "\r\nContent-Type: " + rep.type +
                             "\r\nContent-Length: " + std::to_string(rep.body.size()) +
                             "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (send_all(s, head.data(), head.size())) send_all(s, rep.body.data(), rep.body.size());
    close_sock(s);
}

void nunchi_server::events(intptr_t s, bool audio) {
    const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                             "Connection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\nretry: 1000\n\n";
    if (!send_all(s, head.data(), head.size())) { close_sock(s); return; }
    auto c = std::make_shared<client>();
    c->audio = audio;
    {
        std::lock_guard<std::mutex> l(cm);
        for (auto & h : history) c->out.push_back(h);
        clients.push_back(c);
    }
    auto last = std::chrono::steady_clock::now();
    for (;;) {
        std::deque<std::string> batch;
        {
            std::unique_lock<std::mutex> k(c->m);
            c->cv.wait_for(k, std::chrono::seconds(10), [&] { return !c->out.empty() || c->dead; });
            if (c->dead) break;
            batch.swap(c->out);
        }
        std::string all;
        for (auto & m : batch) all += m;
        if (all.empty() && std::chrono::steady_clock::now() - last >= std::chrono::seconds(10)) all = ": still here\n\n";
        if (!all.empty()) {
            if (!send_all(s, all.data(), all.size())) break;
            last = std::chrono::steady_clock::now();
        }
    }
    {
        std::lock_guard<std::mutex> l(cm);
        for (size_t i = 0; i < clients.size(); i++) if (clients[i] == c) { clients.erase(clients.begin() + i); break; }
    }
    close_sock(s);
}
