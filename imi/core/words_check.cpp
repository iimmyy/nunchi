// Our text -> ids against another tokenizer's answers, text by text.
//   imi-words-check MODEL.gguf TEXTS.bin [THEIRS.txt]
// TEXTS.bin: repeated [uint32 length][UTF-8 bytes]. Prints one line of ids per text; with THEIRS.txt (the same
// format, e.g. llama-server /tokenize), counts the texts that differ and shows the first few. Also decodes our ids
// back and counts round trips that fail.
#include "words.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: imi-words-check MODEL.gguf TEXTS.bin [THEIRS.txt]\n"); return 1; }
    imi_file f;
    if (!f.load(argv[1], false)) return 1;
    imi_words w;
    if (!w.load(f)) return 1;
    std::vector<std::string> texts;
    {
        FILE * in = fopen(argv[2], "rb");
        uint32_t len;
        while (in && fread(&len, 4, 1, in) == 1) { std::string s(len, '\0'); if (fread(s.data(), 1, len, in) != len) break; texts.push_back(s); }
        if (in) fclose(in);
    }
    std::vector<std::string> theirs;
    if (argc > 3) { std::ifstream t(argv[3]); std::string line; while (std::getline(t, line)) theirs.push_back(line); }
    int differ = 0, shown = 0, round = 0;
    size_t n_ids = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        const std::vector<int32_t> ids = w.encode(texts[i]);
        n_ids += ids.size();
        std::ostringstream o;
        for (size_t k = 0; k < ids.size(); k++) o << (k ? " " : "") << ids[k];
        if (argc <= 3) printf("%s\n", o.str().c_str());
        if (i < theirs.size() && theirs[i] != o.str()) {
            differ++;
            if (shown++ < 5) printf("text %zu differs:\n  %s\n  ours   %s\n  theirs %s\n", i, texts[i].substr(0, 200).c_str(), o.str().substr(0, 300).c_str(), theirs[i].substr(0, 300).c_str());
        }
        if (w.text(ids.data(), (int) ids.size()) != texts[i]) round++;
    }
    fprintf(stderr, "imi-words: %zu texts, %zu ids; %d differ from theirs (of %zu); %d decode back wrong\n", texts.size(), n_ids, differ, theirs.size(), round);
    return differ || round;
}
