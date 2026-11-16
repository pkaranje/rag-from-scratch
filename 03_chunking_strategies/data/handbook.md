# Junebug Fleet Operations Handbook

*Internal technical reference for warehouse operators running the Junebug robot fleet.
Fictional document written for a chunking-strategy demo — all model numbers, error codes,
and procedures below are invented.*

## Chapter 1: Fleet Overview

The Junebug fleet consists of autonomous mobile robots (AMRs) that transport shelving pods
between storage zones and human picking stations. Each robot is identified by a unit ID in
the format `JB-####`, printed on a label above the charging port.

A standard warehouse deployment runs between 12 and 80 units depending on floor size. Units
communicate with the central HiveMind coordinator over a dedicated 5GHz mesh network,
separate from the warehouse's general WiFi to avoid congestion.

### 1.1 Hardware Generations

Three hardware generations exist in the field: Gen 1 (2032–2033), Gen 2 (2034–2035), and
Gen 3 (2036–present). Gen 3 units are distinguishable by their teal underglow LED strip,
added purely so operators can visually confirm firmware generation from across a warehouse
floor without checking a screen.

### 1.2 Battery and Charging

Each Junebug carries a 48V lithium-iron-phosphate battery pack rated for approximately
1,800 charge cycles before capacity drops below 80%. A full charge from 10% takes 90
minutes on a standard dock and 35 minutes on a fast-charge dock. Fast-charge docks are
limited to Gen 2 and Gen 3 units; Gen 1 battery controllers will refuse fast-charge current
and fall back to standard charging automatically.

## Chapter 2: Daily Startup Procedure

Every morning before the first picking wave, a floor supervisor must run the startup
checklist:

1. Confirm all docked units show a solid green charge-status light.
2. Open the HiveMind dashboard and verify unit count matches the expected roster.
3. Run the mesh network diagnostic (`hivemind diag --mesh`) and confirm zero dropped nodes.
4. Release units from dock in batches of 5 to avoid a startup current spike.
5. Watch the first lap of each batch to confirm no units are stuck in "confused" state.

### 2.1 Common Startup Failures

If a unit fails to leave its dock, the most common cause is a stale pose estimate from the
previous shutdown. Power-cycling the unit and letting it re-localize against warehouse
floor markers usually resolves this within two minutes. If the unit still won't localize,
the floor marker near its dock may be dirty or obstructed — this accounts for roughly 70%
of localization failures reported to support.

## Chapter 3: Picking Zone Coordination

Picking zones are divided into hexagonal cells, each roughly 4 meters across, to allow
efficient multi-robot pathing without square-grid congestion at intersections. HiveMind
assigns each robot a priority score per cell based on how close it is to completing its
current pod delivery; robots with lower priority yield right-of-way automatically.

### 3.1 Congestion Handling

When more than 6 robots enter overlapping cells within a 10-second window, HiveMind
triggers "congestion mode" for that zone, which temporarily widens the minimum following
distance from 0.5 meters to 1.2 meters and slows all units in the zone to 0.6 m/s. Zones
exit congestion mode automatically once robot density drops below the threshold for 30
consecutive seconds.

### 3.2 Human Safety Zones

Any picking station has a 1.5-meter human safety radius. A robot entering this radius while
a picker badge is detected nearby will automatically reduce speed to walking pace and sound
a soft chime. This is a hardware-level safety behavior — it cannot be disabled from the
HiveMind dashboard, only from a physical technician override switch inside the robot's
service panel.

## Chapter 4: Error Codes

Below is a partial table of error codes operators are expected to recognize without
escalating to engineering.

| Code | Meaning | Typical Fix |
|------|---------|--------------|
| E-101 | Lost localization | Power cycle, re-localize at floor marker |
| E-204 | Battery undervoltage | Return to dock immediately, do not continue picking |
| E-310 | Mesh network timeout | Check for network interference, rerun mesh diagnostic |
| E-415 | Pod weight mismatch | Verify pod isn't overloaded past 40kg rated limit |
| E-522 | Motor stall detected | Clear physical obstruction, restart unit |

### 4.1 Escalation Policy

Any error code above E-500 requires a technician visit within 4 hours per the standard
service agreement. Codes below E-500 may be resolved by trained floor staff following the
fix column above. Any code that recurs three times on the same unit within a single shift
should be escalated regardless of its numeric severity, since repeated low-severity errors
often indicate a developing hardware fault.

## Chapter 5: Software Updates

HiveMind pushes firmware updates to the fleet in staggered batches, never more than 20% of
active units at once, so a bad update can't ground the entire fleet simultaneously. Updates
are staged overnight during the 2 AM to 4 AM maintenance window when picking demand is
lowest.

### 5.1 Rollback Procedure

If a firmware update causes elevated error rates (more than double the 7-day rolling
average for any error code), HiveMind automatically halts the rollout and reverts affected
units to the previous known-good firmware version. Manual rollback can be triggered from
the dashboard's Fleet > Firmware > Rollback panel, which requires supervisor-level
credentials.

## Chapter 6: Maintenance Schedule

Routine maintenance is split into three tiers:

- **Daily**: visual inspection for debris in wheel wells, dock contact cleaning.
- **Weekly**: battery health check, mesh signal strength survey, chassis bolt torque check.
- **Quarterly**: full battery capacity test, wheel bearing replacement if wear exceeds spec,
  firmware audit against the latest approved release.

### 6.1 Wheel Wear Thresholds

Wheel wear is measured by tread depth in millimeters. New wheels ship at 8mm tread depth.
Below 3mm, traction on polished concrete degrades enough to trigger more frequent E-522
motor stall codes, so wheels should be scheduled for replacement at the 3.5mm mark to stay
ahead of failures rather than reacting to them.

## Chapter 7: Decommissioning a Unit

When a unit reaches end of life — typically after 4 to 5 years of continuous service — it
must be decommissioned through the formal process rather than simply powered off and
stored. The battery pack must be discharged to below 30% before long-term storage to avoid
lithium-iron-phosphate cell degradation. The unit ID must be released in HiveMind so it
isn't erroneously flagged as "missing" during the next fleet audit.

### 7.1 Data Wipe Requirement

Every decommissioned unit stores several months of local pose and path-planning logs on an
onboard SSD. Per company policy, this SSD must be wiped using the HiveMind "secure erase"
command before the unit leaves the building, even for units being sent to Acme's own
recycling partner.

## Chapter 8: Frequently Asked Questions

**Q: Can Gen 1 units run the newest HiveMind firmware?**
A: No. Gen 1 units are capped at firmware branch 4.x due to a hardware limitation in their
onboard compute module. Firmware 5.x and later require the Gen 2 compute module at minimum.

**Q: What happens if two robots are assigned the same pod?**
A: HiveMind uses a claim-token system — the first robot to reach the pod's RFID tag wins
the claim, and the second robot is automatically reassigned to the next pod in its queue
within about 2 seconds.

**Q: Why do robots sometimes pause for no visible reason?**
A: This is almost always a congestion-mode yield (see Chapter 3.1) or a human-safety-zone
slowdown (see Chapter 3.2), not a malfunction. Check the HiveMind event log before assuming
a hardware issue.
