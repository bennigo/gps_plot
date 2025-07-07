#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
## -*- coding: iso-8859-15 -*-
This module containes the following functions:

Public functions:
   stdFrame(Ylabel=None, Title=None):
   stdTimesPlot(x, y, Dy, Ylabel=None, Title=None)
   addEvent(eventDict,fig,dstr="%Y%m%d")
   saveFig(fileName,fType,fig)
   toord(yearf):
   fromord(yearf):
   convIcelCh(string)

   Under construction:
       fancytitle(x, y, titlestr, font, ax, **kw):

Private functions:
   __converter(x)

"""

__author__ = "Benedikt G. Ofeigsson <bgo@vedur.is>"
__date__ = "$Date: Feb 2016"
__version__ = "$Revision: 0.1 $"[11:-2]

# sys.setdefaultencoding("utf-8")
#
# Plot functions
#

import datetime
import os
import subprocess
import sys
import warnings
from datetime import timedelta

import gps_parser as cp

import geo_dataread.gps_read as gpsr
import geofunc.geofunc as gf

# import gpstime
import matplotlib as mpl
import matplotlib.image as image

# import matplotlib.colors as mcolors
# import matplotlib.image as image
# from scipy.spatial.distance import euclidean
# extra stuff
# from timesfunc.timesfunc import convGlobktopandas, toDateTime
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geo_dataread.gps_read import convGlobktopandas, toDateTime
from gtimes.timefunc import currDate, currDatetime, currTime, currYearfDate, toDatetimel
from highlight_text import ax_text, fig_text
from matplotlib import transforms


def plotTime(
    sta,
    start=None,
    end=None,
    save=None,
    ylim=[],
    special=None,
    ref="itrf2008",
    figDir="",
    events={},
    fix=False,
    Dir=None,
    tType="TOT",
    uncert=15,
    logo=True,
):
    """
    Plot standard GPS time series North, East up componend

    """
    # for testing
    # print "Station: ",sta
    # print "Start: ", start
    # print "end: ", end
    # print "save: ", save
    # print "ylim: ", ylim
    # print "special: ", special
    # print "ref: ", ref
    # print "events: ", events
    # print "fix: ", fix

    # The backend of matplotlib
    if save == "eps":
        mpl.use("ps")
    elif save == "pdf":
        mpl.use("pdf")
    else:
        mpl.use("WebAgg")
    # Needs to be imported after the backend is defined
    # import matplotlib.pyplot as plt

    fstart = fend = None

    # defining sub-periods to plot
    if start:  # start the plot at
        fstart = currYearfDate(refday=start)
    if end:  # end the plot at
        fend = currYearfDate(refday=end)

    # plottin standard time series special caases
    # filtering and prepearing for ploting
    if special:
        if not end:
            end = currDatetime(-1)
            fend = currYearfDate(refday=end)

        if special == "90d":
            start = currDatetime(days=-91, refday=end)
            fstart = currYearfDate(refday=start)
        if special == "year":
            start = currDatetime(days=-366, refday=end)
            fstart = currYearfDate(refday=start)
        if special == "fixedstart":
            pass
        if special == "full":
            start = None
            fstart = None

    if fix or special:
        pass
    else:  # if fix is None, only plot the extend of the data
        start = end = None

    # creating the graph title
    if ref == "plate":
        plateName = gf.plateFullname(gf.plateDict()[sta])
        refTitle = plateName
    elif ref == "detrend":
        refTitle = "Detrended"
    else:
        refTitle = ref.upper()

    yearf, data, Ddata, offset = gpsr.getData(
        sta, fstart=fstart, fend=fend, ref=ref, Dir=Dir, tType=tType, uncert=uncert
    )
    Pdata = convGlobktopandas(yearf, data, Ddata)
    yearf = pd.to_datetime(toDateTime(yearf))
    datastats = Pdata.describe()
    # print(datastats)
    #    print(datastats.north[3])
    # max( PdataT.index.max(),Pdata.index.max() )

    firstpoint = yearf[0]
    lastpoint = yearf[-1]

    warnp = -1  # at which point we want to label the newest data green
    # title string list [Title, Subtitle]
    Title = makelatexTitle(sta, lastpoint, ref=refTitle, warnp=warnp)
    # print(Title)
    # ploting

    fig = stdTimesPlot(
        yearf, data, Ddata, Title=Title, start=start, end=end, ylim=ylim, warnp=warnp
    )

    if tType == "JOIN":
        yearf, data, Ddata, offset8 = gpsr.getData(
            sta,
            fstart=fstart,
            fend=fend,
            ref=ref,
            Dir=Dir,
            tType="08h",
            uncert=uncert,
            offset=None,
        )
        Pdata8 = convGlobktopandas(yearf, data, Ddata)
        shift = [
            (Pdata8.north - Pdata.north).mean(),
            (Pdata8.east - Pdata.east).mean(),
            (Pdata8.up - Pdata.up).mean(),
        ]
        data = np.array([data[i, :] - shift[i] for i in range(3)])
        yearf = pd.to_datetime(toDateTime(yearf))
        fig = addData(yearf, data, Ddata, fig, markerfacecolor="b", markeredgecolor="b")

    if events:
        addEvent(events, fig)

    # putting a logo
    if logo:
        inpLogo(fig)

    # saving the fiugre to a file
    if save:
        # filename and path
        filend = "-%s" % (ref,)

        if tType != "TOT":
            filend += "-{0:s}".format(tType)

        if special:
            if special == "fixedstart":
                filend += "_since-%s" % (firstpoint.strftime("%Y%m%d"),)
            else:
                filend += "-%s" % special

        else:
            filend += "_%s-%s" % (
                firstpoint.strftime("%Y%m%d"),
                lastpoint.strftime("%Y%m%d"),
            )

        figFile = sta + filend
        fileName = os.path.join(figDir, figFile)
        ##-------------------------------------------
        saveFig(fileName, save, fig)

        # utils.conv2png(fileName+"."+save,logo='/home/bgo/git/gps/postprocessing/matplotlib-based/logos/vi_logos/vi_logo.png')
        # utils.conv2png(fileName+"."+save,logo='/home/bgo/git/gps/gpspostprocessing/matplotlib-based/logos/vi_logos/vi_namelogo.png')
        conv2png(fileName + "." + save, logo=None)
        # utils.convpng2thum(fileName+".png")

    else:
        plt.show()


def stdFrame(Ylabel=None, Title=None):
    """
    Frame for plotting standard GPS time series. takes in Ylabel and Title
    and constructs empty figure with no data.
    input:
        Ylabel,
        Title,

    output:
        fig, Figure object.

    """
    # --- apply custom stuff to the whole fig ---
    mpl.style.use("classic")
    plt.minorticks_on
    plt.gcf().autofmt_xdate(rotation=0, ha="left")
    mpl.rcParams["legend.handlelength"] = 0
    mpl.rcParams["text.usetex"] = True
    # mpl.rcParams['text.latex.unicode'] = True
    mpl.rc("text.latex", preamble=r"\usepackage[utf8]{inputenc}")
    mpl.rc("text.latex", preamble=r"\usepackage{color}")
    # mpl.rc("text.latex", preamble=r"\usepackage[pdftex]{graphicx}")
    # mpl.rc('text.latex', preamble=r'\usepackage[icelandic]{babel}')
    # mpl.rc('text.latex', preamble=r'\usepackage[T1]{fontenc}')
    mpl.rc("font", family="Times New Roman")
    # plt.rcParams["pgf.preamble"] = ( r"\AtBeginDocument{\catcode`\&=12\catcode`\#=12}")

    # constructing a figure with three axes and adding a title
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 20))
    fig.subplots_adjust(hspace=0.1)

    if type(Title) is list:
        plt.suptitle(Title[0], y=0.935, x=0.51)
        axes[0].set_title(Title[1], y=1.01)
    elif type(Title) is str:
        axes[0].set_title(Title)
    else:
        pass

    if Ylabel == None:  #
        Ylabel = ("North [mm]", "East [mm]", "Up [mm]")
        # Ylabel = ("Nordur [mm]","Austur [mm]","Upp [mm]")

    for i in range(3):
        axes[i].set_ylabel(Ylabel[i], fontsize="x-large", labelpad=0)
        # axes[i].axhline(0, xmin=0, xmax=1,color='black') # zero line

    for ax in axes[-1:-4:-1]:
        # tickmarks lables and other and other axis stuff on each axes
        # needs improvement

        # --- X axis ---
        xax = ax.get_xaxis()

        xax.set_tick_params(
            which="minor", reset=True, direction="inout", length=4, top=False
        )
        xax.set_tick_params(
            which="major", reset=True, direction="inout", length=10, top=False
        )

        if ax is axes[0]:  # special case of top axes
            xax.set_tick_params(
                which="major", reset=True, direction="inout", length=10, top=True
            )
            xax.set_tick_params(
                which="minor", reset=True, direction="inout", length=4, top=True
            )
        else:
            ax.spines["top"].set_visible(False)

    return fig, axes


def setXlim(axes, xmin, xmax, start=None, end=None):
    """
    set the extend of the plot
    """

    # the extend of the plot
    if start and end:
        space = (end - start) / 40
    elif start and not end:
        space = (xmax - start) / 40
    elif not start and end:
        space = (end - xmin) / 40
    else:
        space = (xmax - xmin) / 40

    if start:  # start of the plot
        start = start - space
    else:
        start = xmin - space

    if end:  # end of the plot
        end = end + space
    else:
        end = xmax + space
    # ----------------------

    for ax in axes:
        ax.set_xlim(start, end)

    period = end - start

    return period


def setYlim(fig, ymin=[0, 0, 0], ymax=[0, 0, 0], ylim=[]):
    """
    set the extend of the lim
    """

    for i in range(3):
        if len(ylim) == 1:
            fig.axes[i].set_ylim([ymin[i] - ylim[0], ymax[i] + ylim[0]])

        if len(ylim) == 2:
            fig.axes[i].set_ylim(ylim[0], ylim[1])

        if len(ylim) == 3:
            fig.axes[i].set_ylim(ylim[i][0], ylim[i][1])

    return fig


def stdTimesPlot(
    x,
    y,
    Dy,
    Ylabel=None,
    Title=None,
    start=None,
    end=None,
    ylim=[],
    warnp=-1,
    label=None,
):
    """
    plots a three component time series on three different plots
    calls stdFrame to initalize a figure and plots the input data in the frame

    input:
        x
        y
        Dy
        Ylabel
        Title

    output
        fig, Figure object
    """

    # x = pd.to_datetime(toDateTime(x))
    yesterdaybool = x[-1].date() == currDate(warnp)

    # plotting
    fig, axes = stdFrame(Ylabel, Title)
    period = setXlim(axes, x[0], x[-1], start=start, end=end)
    fig = tsTickLabels(fig, axes, period=period)
    # plt.autoscale()

    ymin = [min(y[0, :]), min(y[1, :]), min(y[2, :])]
    ymax = [max(y[0, :]), max(y[1, :]), max(y[2, :])]
    fig = setYlim(fig, ymin=ymin, ymax=ymax, ylim=ylim)

    fig = addData(x, y, Dy, fig, label=label)
    if yesterdaybool:
        fig = addPoints(x[-1], [y[i][-1] for i in range(3)], fig)

    return fig


def addData(
    x,
    y,
    Dy,
    fig,
    ls="none",
    ecolor="grey",
    elinewidth=0.4,
    marker="o",
    markersize=3.5,
    markerfacecolor="r",
    markeredgecolor="r",
    label=None,
):
    """
    Adding data to the plot
    """

    warnings.filterwarnings("ignore")

    for i in range(3):
        fig.axes[i].errorbar(
            x, y[i], yerr=Dy[i], ls=ls, ecolor=ecolor, elinewidth=elinewidth
        )
        fig.axes[i].plot_date(
            x,
            y[i],
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            label=label,
        )

    return fig


def addPoints(
    x,
    y,
    fig,
    marker="o",
    markersize=5.5,
    markerfacecolor="lightgreen",
    markeredgecolor="black",
):
    """
    add a single point to the graph
    """
    for i in range(3):
        fig.axes[i].plot_date(
            x,
            y[i],
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
        )

    return fig


def addEvent(eventDict, fig, dstr="%Y%m%d", **kwargs):
    """
    Adding events
    """

    events = eventDict.keys()
    for event in events:
        if len(eventDict[event]) > 0:
            color = eventDict[event][0]
        # elif  len(eventDict[event]) > 1:
        #    marker = eventDict[event][1]
        # elif  len(eventDict[event]) > 2:
        #    linesyle = eventDict[event][2]
        else:
            color = "r"

        if not isinstance(event, datetime.datetime):
            event = toDatetimel(event, dstr)

        axes = fig.axes
        [ax.axvline(x=event, color=color, zorder=2, **kwargs) for ax in axes]
        # [ ax.axvline(x=event.toordinal(),color=color, zorder=2, **kwargs) for ax in axes ]

    return fig


def saveFig(fileName, fType, fig):
    """
    Save
    """
    plotFile = "%s.%s" % (fileName, fType)
    fig.savefig(plotFile, bbox_inches="tight")

    return fig


#
# Other functions
#


def convLatex(string):
    """
    Converting Icelandic letters to latex (th as þ and d as ð)
    """
    charlist = {
        "Á": r"\'A",
        "á": r"\'a",
        "ð": r"d",
        "Ó": r"\'O",
        "ó": r"\'o",
        "Ö": r"\"O",
        "ö": r"\"o",
        "Ú": r"\'U",
        "ú": r"\'u",
        "Æ": r"{\AE}",
        "æ": r"{\ae}",
        "Í": r"\'I",
        "í": r"\'i",
        "Ý": r"\'Y",
        "ý": r"\'y",
        "Þ": "TH",
        "þ": "\th",
    }
    for key in charlist:
        if key in string:
            string = string.replace(key, charlist[key])

    return string


def convIcelCh(string):
    """
    Converting Icelandic letters
    """
    charlist = {
        "Á": "\\301",
        "á": "\\341",
        "ð": "\\360",
        "Ó": "\\323",
        "ó": "\\363",
        "Ö": "\\326",
        "ö": "\\366",
        "ú": "\\372",
        "æ": "\\346",
        "Í": "\\315",
        "í": "\\355",
        "Ý": "\\335",
        "ý": "\\375",
        "Þ": "\\336",
        "þ": "\\376",
    }
    for key in charlist:
        if key in string:
            string = string.replace(key, charlist[key])

    return string


#
#   ---  labels and text ---
#


def inpLogo(fig, Logo=None):
    """ """

    # asp = 0.6948051948051948
    # asp = 0.93 # Image aspect ratio
    asp = 0.45  # Image aspect ratio
    xlen = 0.07
    ylen = xlen * asp
    xpos = fig.axes[0].get_position().xmin + 0.005
    ypos = (fig.axes[0].get_position().ymax - xlen * asp) - 0.005

    aximage = fig.add_axes(
        [xpos, ypos, xlen, ylen], frameon=False, xticks=[], yticks=[]
    )

    if Logo:
        im = image.imread(Logo)
        aximage.imshow(
            im, aspect="auto", interpolation="quadric", filternorm=1, alpha=0.7
        )


def makelatexTitle(sta, lastData, ref="ITRF2008", warnp=-1):
    """
    Create a Title for standard GPS time series plot using latex package
    """

    config = cp.ConfigParser()
    Title = []

    lastpoint = currDate(refday=lastData, String="Last datapoint: %d %b %Y")
    yesterdaybool = lastData.date() == currDate(warnp)

    # Color of the date value: green if since yesterday, red otherwise
    if yesterdaybool:
        dcolor = "green"
    else:
        dcolor = "red"

    timeofPlot = currTime("(Plot created on %b %d %Y %H:%M %Z)")
    # fetcing the station full name
    try:
        # stName = cp.parseOne(STA,staConFilePath)['station']['name']
        # stName = config.getStationInfo(sta)["name"]
        stName = config.get_config(sta, "station_name")
    except:
        stName = sta
    stName = convLatex(stName)

    # putting the components together
    NameStr = "%s (%s)" % (stName, sta)
    if sta == "hekla":
        NameStr = "Hekla (Summit)"
    refFr = "Reference frame: %s" % ref
    if ref == "Multigas":
        filtering = "Uncorrected"
        refFr = "%s: %s" % (ref, filtering)

    titlestr = [NameStr, refFr, lastpoint, timeofPlot]

    # list of parameter dictionaries
    font = [
        {"color": "black", "size": r"\Huge", "newl": r"\ "},
        {"color": "black", "size": r"\Large", "newl": r""},
        {"color": dcolor, "size": r"\LARGE", "newl": r"\ "},
        {"color": "black", "size": r"\large", "newl": r""},
    ]

    # Creating the latex strings
    tmptitle = ""
    for i in range(2):
        tmptitle += r"%s\textcolor{%s}{%s} %s " % (
            font[i]["size"],
            font[i]["color"],
            titlestr[i],
            font[i]["newl"],
        )
    Title.append(tmptitle)
    tmptitle = ""

    for i in range(2, 4):
        tmptitle += r"%s\textcolor{%s}{%s} %s " % (
            font[i]["size"],
            font[i]["color"],
            titlestr[i],
            font[i]["newl"],
        )
    Title.append(tmptitle)

    return Title


# HighlightText(x=0.5, y=0.5,
#               fontsize=16,
#               ha='center', va='center',
#               s='<This is a title.>\n<and a subtitle>',
#               highlight_textprops=highlight_textprops,
#               fontname='Roboto',
#               ax=ax)


def makeTitle(sta, lastData, ref="ITRF2008", warnp=-1):
    """
    Create a Title for standard GPS time series plot using latex package
    """

    text = (
        "The iris dataset contains 3 species:\n<setosa>, <versicolor>, and <virginica>"
    )
    fig_text(
        s=text,
        x=0.5,
        y=1,
        fontsize=20,
        color="black",
        highlight_textprops=[
            {"color": colors[0], "fontweight": "bold"},
            {"color": colors[1], "fontweight": "bold"},
            {"color": colors[2], "fontweight": "bold"},
        ],
        ha="center",
    )


def tsTickLabels(fig, axes, period=timedelta(90)):
    """ """

    # minorLoc, minorFmt, majorLoc,majorFmt = tsLabels(period)

    # major labels in separate layer
    # -----------
    axes[-1].get_xaxis().set_tick_params(
        which="major", direction="inout", length=14, reset=True, pad=10
    )
    axes[0].get_xaxis().set_tick_params(
        which="major",
        direction="inout",
        length=14,
        reset=True,
        labelbottom=True,
        pad=10,
    )
    axes[1].get_xaxis().set_tick_params(
        which="major",
        direction="inout",
        length=14,
        reset=True,
        labelbottom=True,
        pad=10,
    )
    #
    for ax in axes[-1:-4:-1]:
        # tickmarks lables and other and other axis stuff on each axes
        # needs improvement
        #     # ax.grid(True,linestyle='solid',axis='x')
        # if period < timedelta(2600):
        ax.grid(
            True,
            which="minor",
            axis="x",
        )
        ax.grid(
            True,
            which="major",
            axis="x",
            color="lightgray",
            linestyle="-",
            linewidth=0.8,
        )
        ax.grid(
            True,
            axis="y",
            color="lightgray",
            linestyle="-",
            linewidth=0.8,
        )

        # --- X axis ---
        xax = ax.get_xaxis()
        xax = tslabels(xax, period=period, locator="minor", formater="minor")
        xax = tslabels(xax, period=period, locator="major", formater="major")
        # xax = tslabels(xax, period=period, locator="major", formater=None)

        # if ax is axes[2]:
        #     xax = tslabels(xax, period=period, locator=None, formater="major")

        # xax.set_tick_params(which="minor", reset=True, direction="inout", length=4)
        # xax.set_tick_params(which="major", reset=True, direction="inout", length=15)

        for tick in xax.get_major_ticks():
            tick.label1.set_horizontalalignment("left")

        # for label in xax.get_ticklabels('major')[::]:
        #     label.set_visible(False)
        #     label.set_text("test")
        #     print("text: %s" % label.get_text())
        xax.label.set_horizontalalignment("center")

        # --- Y axis ---
        yax = ax.get_yaxis()
        yax.set_minor_locator(mpl.ticker.AutoMinorLocator())
        yax.set_tick_params(which="minor", direction="in", length=4)
        yax.set_tick_params(which="major", direction="in", length=10)

    return fig


def tslabels(xax, period=timedelta(90), locator=None, formater=None):
    """
    first attempt to make the time scale on the x axis look good
    """

    if period <= timedelta(6):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H:%M")

        majorLoc = mpl.dates.DayLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(12):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator(maxticks=8)
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(18):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 12])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(30):
        minorLoc = mpl.dates.DayLocator()
        minorFmt = mpl.dates.DateFormatter("%d")
        for label in xax.get_ticklabels("major")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.MonthLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %Y")

    elif period <= timedelta(193):
        # minorLoc = mpl.dates.AutoDateLocator(minticks=11, maxticks=24)
        minorLoc = mpl.dates.DayLocator(bymonthday=[1, 7, 14, 21, 28])
        # minorFmt = mpl.dates.ConciseDateFormatter(minorLoc)
        minorFmt = mpl.dates.DateFormatter("%d")

        majorLoc = mpl.dates.MonthLocator(interval=1)
        majorFmt = mpl.dates.DateFormatter("%b %Y")

    elif period <= timedelta(500):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")

        # majorLoc = mpl.dates.MonthLocator(interval=2)
        majorLoc = mpl.dates.YearLocator(1, month=1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(1200):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")
        for label in xax.get_ticklabels("major")[1::2]:
            label.set_visible(False)

        # majorLoc = mpl.dates.MonthLocator(interval=2)
        majorLoc = mpl.dates.YearLocator(1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(2300):
        minorLoc = mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10])
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.YearLocator(1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(5000):
        minorLoc = mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10])
        minorFmt = mpl.dates.DateFormatter("%b")
        for label in xax.get_ticklabels("minor")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator()
        majorFmt = mpl.dates.DateFormatter("%Y")
        for label in xax.get_ticklabels()[1::2]:
            label.set_visible(False)

    elif period <= timedelta(10000):
        minorLoc = mpl.dates.YearLocator(base=1)
        minorFmt = mpl.dates.DateFormatter("")
        for label in xax.get_ticklabels("minor")[:]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator(base=3)
        majorFmt = mpl.dates.DateFormatter("%Y")

    else:
        minorLoc = mpl.dates.YearLocator(1)
        minorFmt = mpl.dates.DateFormatter("")
        # for label in xax.get_ticklabels('minor')[1::2]:
        #    label.set_visible(False)

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.AutoDateFormatter(majorLoc)

    if locator:
        if locator == "minor":
            xax.set_minor_locator(minorLoc)

        else:
            xax.set_major_locator(majorLoc)

    if formater:
        if formater == "minor":
            xax.set_minor_formatter(minorFmt)

        else:
            xax.set_major_formatter(majorFmt)

        return xax

    else:
        return xax


def auto_date_xticks(xax, period=None, locator=None, formatter=None):
    """
    First attempt to make the time scale on the x axis look good

    Example:

    """

    if period <= timedelta(6):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H:%M")

        majorLoc = mpl.dates.DayLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(12):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator(maxticks=8)
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(18):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 12])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(30):
        minorLoc = mpl.dates.DayLocator()
        minorFmt = mpl.dates.DateFormatter("%d")
        for label in xax.get_ticklabels("major")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.MonthLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %Y")

    elif period <= timedelta(193):
        minorLoc = mpl.dates.AutoDateLocator(minticks=15, maxticks=20)
        minorFmt = mpl.dates.DateFormatter("%d")

        majorLoc = mpl.dates.MonthLocator(interval=1)
        majorFmt = mpl.dates.DateFormatter("%b %Y")

    elif period <= timedelta(500):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.YearLocator(1, month=1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(1200):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.MonthLocator((1, 4, 7, 10))
        majorFmt = mpl.dates.DateFormatter("%Y-%m")

    elif period <= timedelta(2300):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.MonthLocator((1, 7))
        majorFmt = mpl.dates.DateFormatter("%Y-%m")

    elif period <= timedelta(5000):
        minorLoc = mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10])
        minorFmt = mpl.dates.DateFormatter("%b")
        for label in xax.get_ticklabels("minor")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator()
        majorFmt = mpl.dates.DateFormatter("%Y")
        for label in xax.get_ticklabels()[1::2]:
            label.set_visible(False)

    elif period <= timedelta(10000):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter()

        majorLoc = mpl.dates.AutoYearLocator()
        majorFmt = mpl.dates.DateFormatter("%Y")
        for label in xax.get_ticklabels()[1::2]:
            label.set_visible(False)

    else:
        minorLoc = mpl.dates.YearLocator(1)
        minorFmt = mpl.dates.DateFormatter("")

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.AutoDateFormatter(majorLoc)

    if locator:
        if locator == "minor":
            xax.set_minor_locator(minorLoc)

        else:
            xax.set_major_locator(majorLoc)

    if formatter:
        if formatter == "minor":
            xax.set_minor_formatter(minorFmt)

        else:
            xax.set_major_formatter(majorFmt)

        return xax

    else:
        return xax


#
#   --- Private functions ---
#


def __converter(x):
    """
    The data extracted are converted to float and
    occational ******* in the data files needs to handled as NAN

    """
    if x == "********":
        return np.nan
    else:
        return float(x)


#
#   --- develpoment ---
#


def fancytitle(x, y, titlestr, font, ax, **kw):
    """
    function under construction intended to implement a fancy title when plotting time series.
    Using stuff like colorcoating the Last datapoint to indicate if yesterdays data was plotted.

    eventually the aim is to implement something in like ax.fancytext(titlestrlist, fontdictlist)
    where each string in titlestrlist is mapped with corresponding fontdict in fontdictlist

    """
    # fancytitle(0.0, 1.07, titlestr, font, axes[0])

    t = ax.transData
    canvas = ax.figure.canvas

    textlist = [None, None, None, None]
    ex = [None, None, None, None]

    axex = ax.get_window_extent()
    axwidth = axex.width

    for i in range(0, len(titlestr), 2):
        textlist[i] = ax.text(
            x, y, " " + titlestr[i] + " ", fontdict=font[i], transform=t
        )
        textlist[i].draw(canvas.get_renderer())
        ex[i] = textlist[i].get_window_extent()
        textlist[i + 1] = ax.text(
            x, y, " " + titlestr[i + 1] + " ", fontdict=font[i + 1], transform=t
        )
        textlist[i + 1].draw(canvas.get_renderer())
        ex[i + 1] = textlist[i + 1].get_window_extent()

        textwidth = ex[i].width + ex[i + 1].width
        x_tmp = (axwidth - textwidth) / 2 / axwidth

        textlist[i].remove()
        textlist[i + 1].remove()

        textlist[i] = ax.text(
            x_tmp, y, " " + titlestr[i] + " ", fontdict=font[i], transform=t
        )
        textlist[i].draw(canvas.get_renderer())
        t = transforms.offset_copy(textlist[i]._transform, x=ex[i].width, units="dots")
        textlist[i + 1] = ax.text(
            x_tmp, y, " " + titlestr[i + 1] + " ", fontdict=font[i + 1], transform=t
        )
        textlist[i + 1].draw(canvas.get_renderer())
        t = transforms.offset_copy(textlist[i]._transform, x=0, units="dots")
        y = y - 0.06

    return textlist


def conv2png(psFile, density=90, logo=None, logoloc="+780+0090"):
    """ """

    fDir = os.path.dirname(psFile)
    fileName, Ftype = os.path.splitext(psFile)

    tmpFile = os.path.join(fDir, "tmp.png")
    pngFile = "%s.%s" % (fileName, "png")

    psCmd = "convert -density %d %s %s " % (density, psFile, tmpFile)
    run_cmd(psCmd)

    trCmd = "convert -trim %s %s " % (tmpFile, pngFile)
    run_cmd(trCmd)
    if os.path.isfile(tmpFile):
        os.remove(tmpFile)

    if logo:
        logoCmd = "composite -compose atop -gravity NorthEast -geometry +{0} -resize '1x1<' -dissolve 70% {1} {2} {3} ".format(
            logoloc, logo, pngFile, tmpFile
        )
        run_cmd(logoCmd)
        os.rename(tmpFile, pngFile)

        # logoCmd = "composite -compose atop -gravity NorthEast -geometry +830+0090 -resize '1x1<' -dissolve 50%" +  " %s %s %s " % (logo, pngFile, tmpFile)


def run_cmd(check_cmd):
    ## Run command

    process = subprocess.Popen(check_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    proc_check_returncode = process.returncode
    proc_check_comm = process.communicate()[0].strip("\n".encode())

    return proc_check_returncode, proc_check_comm


def run_syscmd(check_cmd, p_args):
    """ """
    ## Run command
    currfunc = (
        __name__ + "." + sys._getframe().f_code.co_name + " >>"
    )  # module.object name
    if p_args["debug"]:
        print("%s Starting ..." % currfunc)

    process = subprocess.Popen(check_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    proc_check_returncode = process.returncode
    proc_check_comm = process.communicate()[0].strip("\n")

    # (3)# Make desicions according to output...
    if p_args["debug"]:
        print("%s process.returncode:" % currfunc, proc_check_returncode)
    if p_args["debug"]:
        print(
            currfunc,
            "process.communicate():\n---------------\n",
            proc_check_comm,
            "\n-------------",
        )
    if proc_check_returncode == 0:
        if p_args["debug"]:
            print("%s Command went well.." % currfunc)
    elif proc_check_returncode == 255:
        if p_args["debug"]:
            print("%s Timeout..." % currfunc)
    else:
        if p_args["debug"]:
            print("%s Command failed... " % currfunc)
