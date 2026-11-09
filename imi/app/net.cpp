// nunchi: this computer's addresses on the room's network, best first (a Windows hotspot's own address, then the
// private ranges phones use, then the rest), for the page's address and its QR code.
#include <algorithm>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#endif

static int rank(const std::string & a) {
    if (a == "192.168.137.1") return 0;                       // Windows Mobile Hotspot
    if (a.rfind("192.168.", 0) == 0) return 1;
    if (a.rfind("10.", 0) == 0) return 2;
    if (a.rfind("172.", 0) == 0) {
        const int b = atoi(a.c_str() + 4);
        if (b >= 16 && b <= 31) return 2;
    }
    if (a.rfind("169.254.", 0) == 0) return 9;               // no network yet
    return 5;
}

std::vector<std::string> nunchi_addresses() {
    std::vector<std::string> out;
#ifdef _WIN32
    ULONG len = 16 << 10;
    std::vector<char> buf(len);
    if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr,
                             (IP_ADAPTER_ADDRESSES *) buf.data(), &len) == ERROR_BUFFER_OVERFLOW) {
        buf.resize(len);
        if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr,
                                 (IP_ADAPTER_ADDRESSES *) buf.data(), &len) != NO_ERROR) return out;
    }
    for (auto * a = (IP_ADAPTER_ADDRESSES *) buf.data(); a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
        for (auto * u = a->FirstUnicastAddress; u; u = u->Next) {
            char s[64];
            const sockaddr_in * sin = (const sockaddr_in *) u->Address.lpSockaddr;
            if (inet_ntop(AF_INET, (void *) &sin->sin_addr, s, sizeof s)) out.push_back(s);
        }
    }
#else
    ifaddrs * ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return out;
    for (ifaddrs * i = ifs; i; i = i->ifa_next) {
        if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || (i->ifa_flags & IFF_LOOPBACK) || !(i->ifa_flags & IFF_UP)) continue;
        char s[64];
        if (inet_ntop(AF_INET, &((sockaddr_in *) i->ifa_addr)->sin_addr, s, sizeof s)) out.push_back(s);
    }
    freeifaddrs(ifs);
#endif
    std::stable_sort(out.begin(), out.end(), [](const std::string & a, const std::string & b) { return rank(a) < rank(b); });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
