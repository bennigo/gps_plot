#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
from __future__ import print_function


def main():
    """
    Program to calculete displacements/velocities between to points in time. from a list of stations
    """

    # standard lib
    import argparse
    import sys
    import os
    import traceback

    # special lib
    import timesfunc.timesfunc as gtf
    import cparser as cp

    ref_allow = ["plate", "detrend", "itrf2008"]
    tformat = None

    parser = argparse.ArgumentParser(
        description="return gamit time series in the form of Time NEU DNEU."
    )
    parser.add_argument("Stations", nargs="+", help="List of stations")
    parser.add_argument(
        "--file",
        action="store_true",
        help="write to file Dir/STAT-ref.NEU where Dir is file directory defined by the"
        + " --Dir flag, STAT is the station four letter idendity, ref is defined by the --ref flag ",
    )
    parser.add_argument("--meters", action="store_true", help="print values in meters")
    parser.add_argument(
        "--ref",
        type=str,
        default="plate",
        choices=ref_allow,
        help="Reference frame: defaults to plate, remove plate velocity (plate), Detrend the time series (detrend), No filtering (itrf2008)",
    )
    parser.add_argument(
        "-tf",
        default=tformat,
        type=str,
        help="Format of the output time string If absent, -tf defaults to "
        "%%Y/%%m/%%d %%H:%%M:%%S"
        "." + " Special formating: "
        "yearf"
        " -> decimal year." + " See datetime documentation for formating",
    )
    parser.add_argument(
        "-d",
        "--Dir",
        type=str,
        nargs="?",
        default="",
        const=os.getcwd(),
        help="output directory for files. Defaults to default figDir from cparser",
    )

    args = parser.parse_args()
    print(args, file=sys.stderr)

    stations = args.Stations  # station list
    wfile = args.file
    ref = args.ref
    tformat = args.tf
    meters = not args.meters
    Dir = args.Dir

    if "all" in stations:  # geting a list of all the GPS stations
        stations = [stat["station"]["id"] for stat in cp.Parser().getStationInfo()]

    for sta in stations:
        try:  # Trying to plot
            if wfile:
                outFile = open(
                    os.path.join(Dir, "{0:s}-{1:s}.NEU".format(sta, ref)), "w+"
                )
            else:
                outFile = sys.stdout

            print("wrting to {0:s} ".format(outFile))
            gtf.gamittooneuf(sta, outFile, mm=meters, ref=ref, dstring=tformat)
            # print "Time series of  %s using: %s, %s" % (sta, kwargs['ref'], kwargs['special'])

        except IndexError, e:
            top = traceback.extract_stack()[-1]
            errorstr = "{0:s}:"
            errorstr += ", ".join(
                [
                    type(e).__name__,
                    os.path.basename(top[0]),
                    str(top[1]),
                    "For station {0:s}".format(sta),
                ]
            )
            print(errorst, file=sys.stderr)

        except:
            traceback.print_exc()
            # print >>sys.stderr, top
            print(
                "Unexpected error: {0:s} during processing of {0:s}".format(
                    sys.exc_info()[0], sta
                )
            )


if __name__ == "__main__":
    main()
