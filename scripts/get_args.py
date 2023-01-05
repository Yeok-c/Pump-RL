import sys, getopt


def get_args(argv):
    dra = 0.05
    dra_schedule = 0 # constant
    goal_range_low = 0.3
    goal_range_high = 3.0
    load_vol_range_low = 0
    load_vol_range_high = 2.0
    gamma = 0.92

    opts, args = getopt.getopt(argv,"hi:o:",[
    "dra=","dra_schedule=", 
    "goal_range_low=", "goal_range_high=", 
    "load_vol_range_low=", "load_vol_range_high=",
    "gamma="])

    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -dra <0-0.10> -dra_schedule <1 or 0> \
            -goal_range_low <0.3>, -goal_range_high <3.0>, \
            -load_vol_range_low <0.0>, -load_vol_range_high <2.0>, gamma <0.92>')
            sys.exit()

        elif opt in ("--dra"):
            dra = arg
        
        elif opt in ("--dra_schedule"):
            dra_schedule = arg

        elif opt in ("--goal_range_low"):
            goal_range_low = arg

        elif opt in ("--goal_range_high"):
            goal_range_high = arg

        elif opt in ("--load_vol_range_low"):
            load_vol_range_low = arg

        elif opt in ("--load_vol_range_high"):
            load_vol_range_high = arg

        elif opt in ("--gamma"):
            gamma = arg

    return dra, dra_schedule, goal_range_low, goal_range_high, load_vol_range_low, load_vol_range_high, gamma

if __name__ == "__main__":
    for item in get_args(sys.argv[1:]):
        print(item)
