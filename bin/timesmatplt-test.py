#!/usr/bin/python


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime as dt


import timesmatplt.timesmatplt as tpl
from gtimes.timefunc import TimefromYearf, currTime
import cparser as cp

refrfr = "itrf08"
STA = 'HOFN'
# fetcing the station full name


yearf, data, Ddata = tpl.openGlobkTimes(STA)


# --- fetch full station name
try: 
    stName = cp.Parser().getStationInfo(STA)['station']['name']
except:
    stName = STA

#station full name
stName = tpl.convIcelCh(stName)

# --- to put with the plot title and Ylabel ---
if yearf.any():
    lastpoint = TimefromYearf(yearf.max(),"Last datapoint: %b %d %Y")
else:
    lastpoint = "No data"
timeofPlot = currTime("(Plot created on %b %d %Y %H:%M %Z)")
reference = "Reference frame: %s" % refrfr.upper()

# Title string
#Title =  "0.5 1.13 14 0 4 CM @:20:%s (%s) @:14:%s\n" % (stName, STA, reference)
#Title += "0.5 1.07 14 0 4 CM %s @:11:%s" % (lastpoint, timeofPlot)
Title =  "%s (%s) %s\n%s %s" % (stName, STA, reference, lastpoint, timeofPlot )
#fTitle = r'$\sin %s$', (stName)



# begin and endpoint + sampling
datebeg = TimefromYearf(yearf.min())
dateend = TimefromYearf(yearf.max())
Dtdate = dt.timedelta(days=1)

# from floating point year to floating point ordinal
for i in range(len(yearf)):
    yearf[i] = TimefromYearf(yearf[i],'ordinalf')

# averaging the first seven days
if data.any():
    aver = np.average(data[0:3,0:7],1,weights=1/Ddata[0:3,0:7])
    data = np.array([ data[i,:] - aver[i] for i in range(3)])

# converting to mm
data *= 1000
Ddata *= 1000 

# Filtering a little, removing big outliers
filt = Ddata < 20.0 
filt = np.logical_and(np.logical_and(filt[0,:],filt[1,:]),filt[2,:])
   
yearf = yearf[filt]
data = np.reshape(data[np.array([filt,filt,filt])],(3,-1))
Ddata = np.reshape(Ddata[np.array([filt,filt,filt])],(3,-1))


#tmp = np.logical_and(yearf > TimetoYearf(2011,02,16), yearf < TimetoYearf(2011,03,01))
#tmp1 = np.logical_and(yearf > TimetoYearf(2012,11,25), yearf < TimetoYearf(2013,06,01))
#tmp = np.logical_or(tmp,tmp1)
#tmp1 = np.logical_and(yearf > TimetoYearf(2013,11,04), yearf < TimetoYearf(2013,11,10))
#tmp = np.logical_or(tmp,tmp1)
#tmp1 = np.logical_and(yearf > TimetoYearf(2013,10,1), yearf < TimetoYearf(2013,10,3))
#tmp = np.logical_or(tmp,tmp1)

#index = np.where(tmp)
#yearf = np.delete(yearf, index)
#data = np.delete(data,index,1)
#Ddata = np.delete(Ddata,index,1)

lastpoint = currTime("Last datapoint: %b %d %Y")
timeofPlot = currTime("(Plot created on %b %d %Y %H:%M %Z)")
titlestr = ['Austmannsbunga (AUST)', 'Reference frame: ITRF08', lastpoint, timeofPlot]

font =[{'color':'black', 'size':20}, {'color':'black', 'size':14}, {'color':'green', 'size':14}, {'color':'red', 'size':11}]

Ylabel = ("North [mm]","East [mm]","Up [mm]")

# --- plotting stuff --- 
#
#
#
#-----------------------

# constructing a figure with three axes and adding a title
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11,20))
fig.subplots_adjust(hspace=.1)
axes[0].set_title(Title)
#textlist = tpl.fancytitle(0.0, 1.07, titlestr, font, axes[0])


#constructing the plot 
for i in range(3):
    axes[i].plot_date(yearf,data[i], marker='o', markersize=3.5, markerfacecolor='r', markeredgecolor='r')
    axes[i].errorbar(yearf,data[i],yerr=Ddata[i],ls='none', color='black')
    axes[i].axhline(0,color='black') # zero line
    axes[i].set_ylabel(Ylabel[i], fontsize='x-large', labelpad=0)

# --- apply custom stuff to the whole fig ---
plt.gcf().autofmt_xdate(rotation=0, ha='left')
plt.minorticks_on

# date formatters for the x labels
monthsFmt = mpl.dates.DateFormatter('%b')
dateFmt = mpl.dates.DateFormatter('%Y')

# tick locators
monthsLoc = mpl.dates.MonthLocator(interval=1) # Should actually determain this automatically
yearLoc = mpl.dates.YearLocator()

# major labels in separate layer
axes[-1].get_xaxis().set_tick_params(which='major', pad=19) 

for ax in axes[-1:-4:-1]:
    #tickmarks lables and other and other axis stuff on each axes 
    # needs improvement

    # --- X axis ---
    xax = ax.get_xaxis()

    xax.set_major_formatter(dateFmt)
    xax.set_major_locator(yearLoc)

    xax.set_minor_locator(monthsLoc)
    xax.set_minor_formatter(monthsFmt)

    xax.set_tick_params(which='minor', direction='inout', length=4, top='off')
    xax.set_tick_params(which='major', direction='inout', length=10, top='off')
    
    if ax is axes[0]: # special case of top axes
        xax.set_tick_params(which='major', direction='inout', length=10, top='on')
        xax.set_tick_params(which='minor', direction='inout', length=4, top='on')
    else:
        ax.spines['top'].set_visible(False)
        
    # initiating the minor ticks    
    plt.setp(ax.get_xticklabels(minor=True),visible=False)

    # making only every fourth tick label visible
    # (needs to be dynamic)
    for label in xax.get_minorticklabels()[::4]:
        label.set_visible(True)
        label.set_ha('center')

    

    # --- Y axis ---
    yax = ax.get_yaxis()
    yax.set_minor_locator(mpl.ticker.AutoMinorLocator())
    yax.set_tick_params(which='minor', direction='in', length=4)
    yax.set_tick_params(which='major', direction='in', length=10)


plt.savefig('test.eps')
#fig.show()

