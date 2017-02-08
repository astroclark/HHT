#!/usr/bin/env python
"""
This module contains functions for performing the various steps
of calculating an HHT:

	IMF - returns the first intrinsic mode function (IMF) of a function
	EMD - returns all IMFs of a function
	HT - returns the Hilbert transform
	HHT - returns Hilbert transforms, instantaneous frequencies, 
		and instantaneous amplitudes of all IMFs of a function

See individual function documentation for details.
"""

#This is necessary for HHT
def sgn(x):
	"""The signum function."""
	if x > 0: return 1
	elif x < 0: return -1
	elif x == 0: return 0

def IMF(x,y,threshold=0.0001):
	"""
	IMF(x,y)

	IMF(x,y) returns y-array of the first intrinsic mode function of
	the function given by (x,y).  The convergence threshold can be
	specified with the keyword argument threshold; if none is
	given, it defaults to 0.0001.
	"""

	import extrema
	from numpy import copy
	from scipy import interpolate

	#Initialize the convergence criterion, and loop
	#until it is below threshold
	CC = 2*threshold

	while CC > threshold:
		#Find the extrema points of the curve
		highindx = extrema.maxima(y,withends=True)
		lowindx = extrema.minima(y,withends=True)
		yhigh = y[highindx]
		xhigh = x[highindx]
		ylow = y[lowindx]
		xlow = x[lowindx]

		#Perform a spline interpolation for the upper and lower envelopes
		tckhigh = interpolate.splrep(xhigh,yhigh)
		tcklow = interpolate.splrep(xlow,ylow)

		#Now calculate the envelopes
		yyhigh = interpolate.splev(x,tckhigh)
		yylow = interpolate.splev(x,tcklow)

		#Find the average of the envelopes
		mean = 0.5*(yyhigh + yylow)

		#Subtract the average from the curve
		ynew = y - mean

		#Calculate convergence criterion
		top = sum((abs(y[i]-ynew[i]))**2 for i in range(len(y)))
		bottom = sum((y[i])**2 for i in range(len(y)))
		CC = top/bottom

		#Set y to new curve
		y = copy(ynew)

	return y

def EMD(x,y,N=None,threshold=None):
	"""
	EMD(x,y,N=None,threshold=None)

	EMD decomposes a function given by the arrays (x,y) into
	its IMFs.  Including the keyword N only decomposes into the first N
	IMFs; the keyword threshold sets the threshold on the convergence
	parameter (if none is specified, it will default to the default set
	in the function IMF).  A two-tuple (IMFs,R) will be returned; IMFs is
	a list containing the IMFs, with IMFs[0] being the original function;
	R is the residue.
	"""

	from numpy import copy
	
	#Initialize the IMF list and the residue R
	IMFs = [y]
	R = copy(y)

	if N:
		#Loop N times
		for i in range(N):

			#Find the next IMF
			if threshold:
				ynew = IMF(x,R,threshold=threshold)
			else:
				ynew = IMF(x,R)

			#Add ynew to IMF list
			IMFs.append(ynew)

			#Subtract ynew from previous residue to get new residue
			R -= IMFs[i+1]

	else:
		#Start looping
		i = 0
		while True:
			try:

				#Find the next IMF
				if threshold:
					ynew = IMF(x,R,threshold=threshold)
				else:
					ynew = IMF(x,R)

				#Add ynew to IMF list
				IMFs.append(ynew)

				#Subtract ynew from previous residue to get new data
				R -= IMFs[i+1]

				#Advance index
				i += 1

			#If error, stop the loop
			except TypeError:
				break

	return (IMFs,R)

def HT(y):
	"""
	HT(y)

	Computes the Hilbert transform of the array y using the 
	frequency-domain approach, as in
	http://library2.usask.ca/theses/available/etd-09272006-135609/unrestricted/XianglingWang.pdf, pages 36-37

	For a good transform, the following requirements must be satisfied:
	1. The numbers of zeros and local extrema in y must differ by at most one.
	2. y must be symmetric about zero (i.e. the mean value of the envelopes 
		defined the by the local maxima and minima of y must be zero).
	"""

	from scipy import fft, ifft
	from numpy import zeros_like

	#Take the Fourier transform of the function
	ffty = fft(y)

	#Write as the FFT of the Hilbert transform
	Y = zeros_like(ffty)
	N = len(ffty)
	for i in range(-N/2+1,N/2+1):
		Y[i] = -1j*sgn(i)*ffty[i]

	#Take the inverse Fourier transform
	HT = ifft(Y)

	return HT

def HHT(x,y,N=None,threshold=None):
	"""
	HHT(x,y,N=None,threshold=None)

	Computes the Hilbert-Huang Transform of the function given
	by the arrays x,y.  Returns a tuple (IMFs,R,amp,freq) where:

		IMFs is a list of the IMFs of the initial function,
			with IMFs[0] being the initial function
		R is the redidue from the IMF decomposition
		amp is a list containing the instantaneous amplitude
			arrays of each IMF
		freq is a list containing the instantaneous (angular) 
			frequency arrays of each IMF

	Including the keyword N will only yield the first N IMFs; the
	keyword	threshold sets the threshold for convergence in the 
	individual IMF decompositions (default value is set in the function IMF).
	If N=0, no IMFs will be calculated; only the tuple (amp,freq)
	for the given data will be returned.
	"""

	from scipy import interpolate
	from math import pi

	#Calculate the IMFs
	if N != 0:
		(IMFs,R) = EMD(x,y,N=N,threshold=threshold)
	else:
		IMFs = [y]

	#Initialize
	amp = []
	freq = []

	#Loop over all IMFs
	import numpy as np
	for mode in IMFs:

		#Calculate Hilbert transform
		htrans = HT(mode)

		#Find spline interpolations for mode and htrans, to find derivatives
		tckmode = interpolate.splrep(x,mode)
		tckhtrans = interpolate.splrep(x,htrans)

		#Now find the derivatives
		ddx_mode = interpolate.splev(x,tckmode,der=1)
		ddx_htrans = interpolate.splev(x,tckhtrans,der=1)

		#Find amplitude and angular frequency of the function mode + 1j*htrans
		a = abs(mode+1j*htrans)
		omega = (1/a**2)*(mode*ddx_htrans - ddx_mode*htrans)

		#Add to lists
		amp.append(a)
		freq.append(omega/(2*pi))
	
	if N != 0:
		return (IMFs,R,amp,freq)
	else:
		return (amp[0],freq[0])

def plotHHT(time,amp,freq,lowFreq=0.0,highFreq=None,figname=None):

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')     # This lets us make plots without a display.
    from matplotlib import pyplot as plt

    fig_width_pt = 600  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    matplotlib.rcParams.update(
            {'axes.labelsize': 16, 
                'text.fontsize':   16, 
                'legend.fontsize': 16, 
                'xtick.labelsize': 16, 
                'ytick.labelsize': 16, 
                'text.usetex':
                True,
                'figure.figsize':
                fig_size,
                'font.family':
                "serif",
                'font.serif':
                ["Times"],
                'savefig.dpi':
                200,
                'xtick.major.size':8,
                'xtick.minor.size':4,
                'ytick.major.size':8,
                'ytick.minor.size':4
                })  

    # cut out LF
    time=time[freq>lowFreq]
    amp=amp[freq>lowFreq]
    freq=freq[freq>lowFreq]

    # cut out low amplitudes
#   cutOffAmp=max(amp)/np.exp(3)
#   time=time[amp>cutOffAmp]
#   freq=freq[amp>cutOffAmp]
#   amp=amp[amp>cutOffAmp]

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.125)

    ax1.plot(time, amp, 'k-',label='IA')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Instantaneous amplitude, IA')
    ax1.minorticks_on()
    #ax1.set_xlim(0.5,1.7)
    #ax1.set_ylim(0,1.5e-20)


    ax2 = ax1.twinx()
    if highFreq is None:
        highFreq=max(ax1.get_ylim())
        
#    ax2.set_ylim(lowFreq,highFreq)

    ax2.plot(time, freq, 'm-', label='IF')

    #ax2.set_xlim(0.5,1.7)
    #ax2.set_ylim(0,100)


    # dummy point for legend
    xlims=ax2.get_xlim()
    ylims=ax2.get_ylim()
    ax2.plot(-10,-10, 'k-', label='IA')
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.minorticks_on()

    ax2.set_ylabel('Instantaneous frequency, IF [Hz]')
    ax2.legend(loc='upper left')

    if figname is None:
        fig.savefig('HilbertSpectrum.png')
    else:
        fig.savefig(figname+'.eps')
    #plt.close(fig)


def matHHT(time,amp,freq):

    import numpy as np
    import matplotlib.pyplot as plt

    freq=freq.real

    freqAxis=freq#np.arange(0.0,max(freq),10)
    M=np.zeros(shape=(len(freqAxis),len(time)))+1e-50
    for i,t in enumerate(time):
        print i,len(time)

        if freq[i]>min(freqAxis) and freq[i]<max(freqAxis):

            # find frequency bin at time t
            fbin=abs(freqAxis-freq[i]).argmin()

            # find amplitude at time t, frequency f
            M[fbin,i] = amp[i]

    # plot it with:
    """
    figure()
    imshow(log10(M), aspect='auto', origin='lower', interpolation='nearest',
            cmap=cm.hot, 
            extent=[min(time),max(time),min(freq),max(freq)])
    """

    return M






