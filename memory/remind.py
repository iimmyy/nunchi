"""Does his own surprise notice a changed fact? (step 9 B, first test; Immy 2026-11-08: "Let's anchor it in how a
human decides on memories")

Wahlheim & Jacoby (2013, "Remembering change"): when a change is noticed while the new fact is heard, the old fact
is embedded in the new memory with its order kept (a recursive reminding), and later the newest wins. When it isn't
noticed, the old one wins (proactive interference). The human trigger is prediction error. Here each later chat is
read twice in chat A (shrinktest's recording read): alone, and with the earlier chats playing first (the linked
recording). Per word piece, jump = his surprise with the old chats in mind minus his surprise alone. The bank marks
the turns that hold the answer (has_answer); the report lists the biggest jumps of each later chat in context and
whether they land in a marked turn. The grade is read by hand: a jump counts only if it sits on the changed value.

The model is untouched and nothing is asked of it: this only listens to how surprised he is.
"""
import argparse
import json
from pathlib import Path

import shrinktest as st


def session_text(turns):
    """LongMemEval's "nl" text of a session (same as shrinktest.nl) plus each turn's character range"""
    out, spans = "", []
    for t in turns:
        head = f"\n\n{t['role']}: "
        a = len(out) + len(head)
        out += head + t["content"].strip()
        spans.append(dict(a=a, b=len(out), role=t["role"], answer=bool(t.get("has_answer"))))
    assert out == st.nl(turns)
    return out, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="longmemeval_oracle.json")
    ap.add_argument("--qids", required=True, help="question ids, one per line")
    ap.add_argument("--qtype", default="knowledge-update")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = [l.strip() for l in open(a.qids) if l.strip()]
    bank = {x["question_id"]: x for x in json.load(open(a.data))}
    todo = [bank[q] for q in want if bank[q]["question_type"] == a.qtype and not q.endswith("_abs")]
    rig = st.Rig(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep, raw = open(out / "report.txt", "w"), open(out / "jumps.jsonl", "w")
    tally = dict(sessions=0, top1=0, top5=0, top20=0)
    for x in todo:
        ses = sorted(zip(x["haystack_dates"], x["haystack_session_ids"], x["haystack_sessions"]), key=lambda z: z[0])
        sessions, spans = [], []
        for d, sid, turns in ses:
            text, sp = session_text(turns)
            sessions.append(dict(sid=sid, date=d, text=text))
            spans.append(sp)
        rep.write(f"\n######## {x['question_id']} | Q: {x['question']} | GOLD: {x['answer']}\n")
        for si in range(len(sessions)):
            for t in spans[si]:
                if t["answer"]:
                    rep.write(f"   S{si + 1} {sessions[si]['date']} marked {t['role']}: "
                              f"{sessions[si]['text'][t['a']:t['b']][:220]!r}\n")
        for si in range(1, len(sessions)):
            s = sessions[si]
            enc = rig.tok(s["text"], add_special_tokens=False, return_offsets_mapping=True)
            span = enc["input_ids"]
            assert span == rig.ids(s["text"])
            _, alone = rig.capture(st.chat_a(rig, s), span)
            _, linked = rig.capture(st.chat_a_linked(rig, sessions, si), span)
            jump = [l - o for l, o in zip(linked, alone)]
            turn_of = []
            for (c0, c1) in enc["offset_mapping"]:
                k = next((i for i, t in enumerate(spans[si]) if t["a"] <= c0 < t["b"]), None)
                turn_of.append(k)
            order = sorted(range(len(jump)), key=lambda j: -jump[j])
            rank = {j: r + 1 for r, j in enumerate(order)}
            marked = [j for j in range(len(jump)) if turn_of[j] is not None and spans[si][turn_of[j]]["answer"]]
            best = min((rank[j] for j in marked), default=None)
            tally["sessions"] += 1
            for name, cut in (("top1", 1), ("top5", 5), ("top20", 20)):
                tally[name] += best is not None and best <= cut
            rep.write(f"\n  later chat S{si + 1} ({len(span)} pieces, {len(marked)} in marked turns): best rank of a "
                      f"marked piece = {best}\n")
            for r, j in enumerate(order[:a.top]):
                k = turn_of[j]
                where = "-" if k is None else f"{spans[si][k]['role']}{' MARKED' if spans[si][k]['answer'] else ''}"
                left = rig.tok.decode(span[max(0, j - 10):j])
                mid = rig.tok.decode(span[j:j + 1])
                right = rig.tok.decode(span[j + 1:j + 5])
                rep.write(f"   #{r + 1} jump {jump[j]:+.2f} (alone {alone[j]:.2f}, linked {linked[j]:.2f}) [{where}] "
                          f"{(left + '[[' + mid + ']]' + right)!r}\n")
            raw.write(json.dumps(dict(qid=x["question_id"], session=si + 1, alone=alone, linked=linked,
                                      turn_of=turn_of, marked=[spans[si][k]["answer"] if k is not None else False
                                                               for k in turn_of])) + "\n")
            rep.flush()
            raw.flush()
    rep.write(f"\nlater chats {tally['sessions']}: a marked piece is the biggest jump in {tally['top1']}, in the top 5 "
              f"in {tally['top5']}, in the top 20 in {tally['top20']}\n")
    rep.close()
    raw.close()
    print(open(out / "report.txt").read().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
