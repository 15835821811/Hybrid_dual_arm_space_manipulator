# Independent discrete braking software preflight

PASS_SOFTWARE_PREFLIGHT_ONLY: no blocking source or arithmetic finding.

Read-only source comparison, AST-isolated NumPy arithmetic, frozen contract metadata comparison and terminal file byte hashing only. No robot modules were imported, no model instantiated, and no FK, distance, coupled or native evaluation occurred. No old raw trajectory arithmetic was repeated.

The right-endpoint ramp displacement is 0.009 v + 0.011 u. Independent explicit ten-substep arithmetic checks the returned feasible lower endpoint, midpoint and upper endpoint against every original bound throughout emergency stopping, including both +/-0.02 tracking-error extremes. The interval tightens the original box and does not mutate inputs. Detailed counts and reviewed source hashes are in preflight_review.json.

Execution, reference, evaluation gates, budget, experiment and visual source ASTs remain identical. Protocol values are unchanged except version. The frozen same-scene terminal bytes, original seven waypoint positions, rotations and timing, static goals and satellite velocity are preserved.

Recorded original components, braking endpoints, final intersection, limiting future times, errors and actual commands support later algebraic reconstruction. Current tracking prerequisite failure or an empty intersection fails closed.

The full-cycle JSON includes 600 seconds of read-only media plus previous read-only overhead beyond the plan worker subtotal. The parent supplied this accounting clarification; old cost raw files were not re-analyzed.

This is a conditional joint/reference braking feasibility software check. It does not establish future geometry, torque, task tracking, successful completion or repair effectiveness. Actual execution and full independent replay remain required.
