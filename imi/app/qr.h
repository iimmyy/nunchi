// nunchi: QR codes for the screen page (join the Wi-Fi, open the page). Byte mode, error correction level M, the
// smallest version from 1 to 10 that fits (up to 213 bytes), the mask with the lowest penalty, as the QR standard
// (ISO/IEC 18004) lays it out.
#pragma once
#include <string>
#include <vector>

// the code's modules, size x size, true = dark; empty if the text doesn't fit
std::vector<std::vector<bool>> nunchi_qr(const std::string & text);
// as an SVG with a 4-module quiet zone
std::string nunchi_qr_svg(const std::string & text);
