// nunchi: a small web server for the room's own network. It serves the pages, sends live events to every phone
// (server-sent events: one long answer per phone that new lines are written into), and takes the sound the screen page
// hears. One thread per connection; each phone's events go out on its own thread, so a slow phone holds up nobody.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct nunchi_request {
    std::string method, path, query, body;
    std::string q(const std::string & key) const;   // a query value ("" when absent)
};
struct nunchi_reply {
    int code = 200;
    std::string type = "text/html; charset=utf-8";
    std::string body;
};

struct nunchi_server {
    // every request but GET /events
    std::function<nunchi_reply(const nunchi_request &)> handle;
    bool start(int port, std::string & err);
    void stop();
    // an event to every phone listening; keep: also given to phones that join later (the last 80 kept); audio: only
    // to phones listening to the voice (GET /events?audio=1)
    void send(const std::string & event, const std::string & json, bool keep = false, bool audio = false);
    int listeners() const;
    int voice_listeners() const;
    ~nunchi_server();

private:
    struct client {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::string> out;
        bool audio = false, dead = false;
    };
    std::atomic<bool> running{ false };
    intptr_t lsock = -1;
    std::thread acceptor;
    mutable std::mutex cm;
    std::vector<std::shared_ptr<client>> clients;
    std::deque<std::string> history;
    void serve(intptr_t s);
    void events(intptr_t s, bool audio);
};

std::string nunchi_json(const std::string & s);   // a JSON string, quotes included
std::string nunchi_base64(const uint8_t * p, size_t n);
