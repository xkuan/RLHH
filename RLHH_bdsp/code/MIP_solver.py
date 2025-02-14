
"""This model implements a bus driver scheduling problem.

Constraints:
- max driving time per driver <= 9h
- max working time per driver <= 12h
- min working time per driver >= 6.5h (soft)
- 30 min break after each 4h of driving time per driver
- 10 min preparation time before the first shift
- 15 min cleaning time after the last shift
- 2 min waiting time after each shift for passenger boarding and alighting
"""

import collections
from ortools.sat.python import cp_model
from data import VCSPDataSet


def bus_driver_scheduling(shifts, max_num_drivers, print_soution=False, time_limit=3600):
    """Optimize the bus driver scheduling problem.
    Returns:
      The objective value of the model.
    """
    num_shifts = len(shifts)

    # All durations are in minutes.
    max_driving_time = 480  # 8 hours.
    max_driving_time_without_pauses = 240  # 4 hours
    min_pause_after_4h = 30
    min_delay_between_shifts = 2
    max_working_time = 720
    min_working_time = 0
    setup_time = 10
    cleanup_time = 15

    # Computed data.
    total_driving_time = sum(shift[5] for shift in shifts)

    num_drivers = max_num_drivers
    min_start_time = min(shift[3] for shift in shifts)
    max_end_time = max(shift[4] for shift in shifts)

    # print('Bus driver scheduling')
    # print('  num shifts =', num_shifts)
    # print('  num drivers =', num_drivers)
    # print('  min start time =', min_start_time)
    # print('  max end time =', max_end_time)

    model = cp_model.CpModel()

    # For each driver and each shift, we store:
    #   - the total driving time including this shift
    #   - the acrued driving time since the last 30 minute break
    # Special arcs have the following effect:
    #   - 'from source to shift' sets the starting time and accumulate the first
    #      shift
    #   - 'from shift to end' sets the ending time, and fill the driving_times
    #      variable
    # Arcs between two shifts have the following impact
    #   - add the duration of the shift to the total driving time
    #   - reset the accumulated driving time if the distance between the two
    #     shifts is more than 30 minutes, add the duration of the shift to the
    #     accumulated driving time since the last break otherwise

    # Per (driver, shift) info (driving time, performed, driving time since break)
    total_driving = {}
    no_break_driving = {}
    performed = {}
    starting_shifts = {}

    # Per driver info (start, end, driving times, is working)
    start_times = []
    end_times = []
    driving_times = []
    # working_drivers = []
    working_times = []

    # Weighted objective
    delay_literals = []
    delay_weights = []

    # Used to propagate more between drivers
    shared_incoming_literals = collections.defaultdict(list)
    shared_outgoing_literals = collections.defaultdict(list)

    for d in range(num_drivers):
        start_times.append(
            model.NewIntVar(min_start_time - setup_time, max_end_time,
                            'start_%i' % d))
        end_times.append(
            model.NewIntVar(min_start_time, max_end_time + cleanup_time,
                            'end_%i' % d))
        driving_times.append(
            model.NewIntVar(0, max_driving_time, 'driving_%i' % d))
        working_times.append(
            model.NewIntVar(0, max_working_time, 'working_times_%i' % d))

        incoming_literals = collections.defaultdict(list)
        outgoing_literals = collections.defaultdict(list)
        outgoing_source_literals = []
        incoming_sink_literals = []

        # Create all the shift variables before iterating on the transitions between these shifts.
        for s in range(num_shifts):
            total_driving[d, s] = model.NewIntVar(0, max_driving_time,
                                                  'dr_%i_%i' % (d, s))
            no_break_driving[d, s] = model.NewIntVar(
                0, max_driving_time_without_pauses, 'mdr_%i_%i' % (d, s))
            performed[d, s] = model.NewBoolVar('performed_%i_%i' % (d, s))

        for s in range(num_shifts):
            shift = shifts[s]
            duration = shift[5]

            # Arc from source to shift.
            #    - set the start time of the driver
            #    - increase driving time and driving time since break
            source_lit = model.NewBoolVar('%i from source to %i' % (d, s))
            outgoing_source_literals.append(source_lit)
            incoming_literals[s].append(source_lit)
            shared_incoming_literals[s].append(source_lit)
            model.Add(start_times[d] == shift[3] -
                      setup_time).OnlyEnforceIf(source_lit)
            model.Add(total_driving[d, s] == duration).OnlyEnforceIf(source_lit)
            model.Add(no_break_driving[d, s] == duration).OnlyEnforceIf(source_lit)
            starting_shifts[d, s] = source_lit

            # Arc from shift to sink
            #    - set the end time of the driver
            #    - set the driving times of the driver
            sink_lit = model.NewBoolVar('%i from %i to sink' % (d, s))
            outgoing_literals[s].append(sink_lit)
            shared_outgoing_literals[s].append(sink_lit)
            incoming_sink_literals.append(sink_lit)
            model.Add(end_times[d] == shift[4] +
                      cleanup_time).OnlyEnforceIf(sink_lit)
            model.Add(
                driving_times[d] == total_driving[d, s]).OnlyEnforceIf(sink_lit)

            # Node not performed
            #    - set both driving times to 0
            #    - add a looping arc on the node
            model.Add(total_driving[d,
                                    s] == 0).OnlyEnforceIf(performed[d,
                                                                     s].Not())
            model.Add(no_break_driving[d, s] == 0).OnlyEnforceIf(
                performed[d, s].Not())
            incoming_literals[s].append(performed[d, s].Not())
            outgoing_literals[s].append(performed[d, s].Not())
            # Not adding to the shared lists, because, globally, each node will have
            # one incoming literal, and one outgoing literal.

            # Node performed:
            #    - add upper bound on start_time
            #    - add lower bound on end_times
            model.Add(start_times[d] <= shift[3] - setup_time).OnlyEnforceIf(
                performed[d, s])
            model.Add(end_times[d] >= shift[4] + cleanup_time).OnlyEnforceIf(
                performed[d, s])

            for o in range(num_shifts):
                other = shifts[o]
                delay = other[3] - shift[4]
                if delay < min_delay_between_shifts:
                    continue
                lit = model.NewBoolVar('%i from %i to %i' % (d, s, o))

                # Increase driving time
                model.Add(total_driving[d, o] == total_driving[d, s] +
                          other[5]).OnlyEnforceIf(lit)

                # Increase no_break_driving or reset it to 0 depending on the delay
                if delay >= min_pause_after_4h:
                    model.Add(
                        no_break_driving[d, o] == other[5]).OnlyEnforceIf(lit)
                else:
                    model.Add(no_break_driving[d, o] == no_break_driving[d, s] +
                              other[5]).OnlyEnforceIf(lit)

                # Add arc
                outgoing_literals[s].append(lit)
                shared_outgoing_literals[s].append(lit)
                incoming_literals[o].append(lit)
                shared_incoming_literals[o].append(lit)

                # Cost part
                delay_literals.append(lit)
                delay_weights.append(delay)

        model.Add(working_times[d] == end_times[d] - start_times[d])

        # Working time constraints
        model.Add(working_times[d] >= min_working_time)

        # Create circuit constraint.
        model.AddExactlyOne(outgoing_source_literals)
        for s in range(num_shifts):
            model.AddExactlyOne(outgoing_literals[s])
            model.AddExactlyOne(incoming_literals[s])
        model.AddExactlyOne(incoming_sink_literals)

    # Each shift is covered.
    for s in range(num_shifts):
        model.AddExactlyOne(performed[d, s] for d in range(num_drivers))
        # Globally, each node has one incoming and one outgoing literal
        model.AddExactlyOne(shared_incoming_literals[s])
        model.AddExactlyOne(shared_outgoing_literals[s])

    # Symmetry breaking

    # The first 3 shifts must be performed by 3 different drivers.
    # Let's assign them to the first 3 drivers in sequence
    model.Add(starting_shifts[0, 0] == 1)
    model.Add(starting_shifts[1, 1] == 1)
    model.Add(starting_shifts[2, 2] == 1)

    # Redundant constraints: sum of driving times = sum of shift driving times
    model.Add(cp_model.LinearExpr.Sum(driving_times) == total_driving_time)
    model.Add(
        cp_model.LinearExpr.Sum(working_times) == total_driving_time +
        num_drivers * (setup_time + cleanup_time) +
        cp_model.LinearExpr.WeightedSum(delay_literals, delay_weights))


    # Minimize the sum of delays between tasks, which in turns minimize the
    # sum of working times as the total driving time is fixed
    model.Minimize(
        cp_model.LinearExpr.WeightedSum(delay_literals, delay_weights))

    # Solve model.
    solver = cp_model.CpSolver()
    # solution_printer = cp_model.ObjectiveSolutionPrinter()
    solver.parameters.max_time_in_seconds = time_limit

    # status = solver.Solve(model, solution_callback=solution_printer)
    status = solver.Solve(model)

    if status == cp_model.UNKNOWN:
        print("MIP_solver: Time Out!")
    # if status == cp_model.INFEASIBLE:
    #     return -1

    if status == cp_model.OPTIMAL and print_soution:
        # Display solution
        for d in range(num_drivers):
            route = ["Source"]
            print('Driver %i: ' % (d + 1))
            # print('  total driving time =', solver.Value(driving_times[d]))
            print('  working time =', solver.Value(working_times[d]))

            first = True
            for s in range(num_shifts):
                shift = shifts[s]

                if not solver.BooleanValue(performed[d, s]):
                    continue
                # route.append(shift[0]+1)
            # route.append("Sink")
            # print(route)


                if solver.Value(no_break_driving[d, s]) == shift[5] and not first:
                    print('    **break**')
                print('    shift ', shift[0], ':', shift[1], '-', shift[2])
                first = False

    return int(solver.ObjectiveValue()) + num_drivers * (setup_time + cleanup_time) + total_driving_time


def solve_bus_driver_scheduling():
    """Optimize the bus driver allocation in two passes."""
    print('----------- first pass: minimize the number of drivers')
    num_drivers = bus_driver_scheduling(True, -1)
    if num_drivers == -1:
        print('no solution found, skipping the final step')
    else:
        print('----------- second pass: minimize the sum of working times')
        bus_driver_scheduling(False, num_drivers)



if __name__ == '__main__':
    data_path = '../data/shift_50_01.csv'
    data = VCSPDataSet(path=data_path)
    objval = bus_driver_scheduling(data.shifts, 12, True)
    print("Total Cost: ", objval)