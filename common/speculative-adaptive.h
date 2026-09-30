#pragma once

#include <algorithm>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine with a climb counter and a weighted drop-pressure
// accumulator. The depth N climbs one step after N_CLIMB(N) consecutive verifies
// that accepted every drafted token. The climb cost is low at the floor and at
// depth, high in the middle: 2 at depth 1, 4 at depth 2, 6 at depth 3, then
// 5/4/3/2 from depth 4 upward. Getting from the floor to depth 3 needs only 6
// full accepts, but pushing past 3 (where prose acceptance collapses) costs 6
// full accepts of 3-token drafts, which predictable content clears quickly and
// marginal content never does. Any miss adds (n_draft - n_accepted) to a
// drop-pressure accumulator; when it reaches depth * 5 the depth drops one step
// and the pressure resets. A near miss (n_draft-1) adds 1, a total miss adds
// n_draft, so high depths fall quickly while low depths hold. The drop budget
// scales with depth but never drops below 20, so shallow depths shed bad content
// quickly without collapsing to the floor on a few bad rounds; deep depths hold
// a little longer. At the floor no pressure accumulates at all. The depth starts
// at the floor max(1, --spec-draft-n-min-adaptive) and stays in
// [floor, n_max]; --spec-draft-n-max bounds the upper end of the adaptive
// range.
//
// Context-aware adjustment: at deeper KV contexts, verification is more
// expensive (longer attention scan), so the controller becomes more
// conservative: climb thresholds increase and drop pressure decreases,
// causing the depth to settle lower at high context depths. This mirrors
// the draft-cost awareness in gufo's adaptive controller.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    // Context depth factor: 1.0 at shallow context, up to 2.0 at very deep context.
    // Scales climb thresholds up (more conservative) and drop pressure down
    // (faster retreat) as context grows.
    static float context_factor(int ctx_len) {
        if (ctx_len <= 8192) return 1.0f;
        if (ctx_len <= 32768) return 1.25f;
        if (ctx_len <= 65536) return 1.5f;
        if (ctx_len <= 131072) return 1.75f;
        return 2.0f;
    }

    // consecutive full accepts needed to climb one step from depth N;
    // scaled by context factor: at deep contexts, climbing is harder
    static int climb_threshold(int depth, int ctx_len = 0) {
        int base;
        switch (depth) {
            case 1: base = 2; break;
            case 2: base = 4; break;
            case 3: base = 6; break;
            case 4: base = 5; break;
            case 5: base = 4; break;
            case 6: base = 3; break;
            default: base = 2; break; // depth >= 7
        }
        if (ctx_len > 0) {
            base = static_cast<int>(base * context_factor(ctx_len) + 0.5f);
        }
        return base;
    }

    // accumulated (n_draft - n_accepted) needed to drop one step from depth N;
    // scaled by depth, with a floor so shallow depths do not collapse too fast;
    // at deep contexts the budget shrinks (faster retreat from unprofitable depths)
    static int drop_pressure(int depth, int ctx_len = 0) {
        int base = std::max(depth * 5, 20);
        if (ctx_len > 0) {
            // reduce the budget at deep context: divide by context factor
            base = static_cast<int>(base / context_factor(ctx_len) + 0.5f);
            base = std::max(base, 10); // never below 10
        }
        return base;
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    // feed one verification result: n_draft is the number of tokens this
    // implementation drafted, n_accepted the number the target accepted,
    // ctx_len is the current KV cache depth (0 = unknown, no context scaling)
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive, int ctx_len = 0) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (n_accepted == n_draft) {
            n_drop = 0;

            // full acceptance: reset the drop pressure, accumulate the climb streak
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur, ctx_len)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            // any miss adds (n_draft - n_accepted) to the drop pressure; drop one
            // step when the accumulated pressure reaches the depth-scaled budget
            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur, ctx_len)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }

    // current draft depth
    int depth() const { return n_cur; }
};
