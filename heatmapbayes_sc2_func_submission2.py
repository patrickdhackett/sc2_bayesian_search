# -*- coding: utf-8 -*-
"""
Appendix A

bayesiansearch_sc2.py
Patrick Hackett
"""
import math
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import vectorize
import pandas as pd
import scipy
from scipy import spatial
from scipy.stats import beta
import os

#os.makedirs("C:/Files/Data/eeg")
#find data
os.chdir("C:/Users/Patrick/Dropbox/Files/Coding/github/sc2_bayesian_search")
#%ls

### FUNCTIONS
##Make a minimap as background of plot from image
def minimap(img): #mpimg.imread('*.png')
    plt.figure(figsize=(6,6))
    plt.imshow(img, origin = 'upper')
    plt.yticks([])
    plt.xticks([])
    plt.ylim([256,0])
    plt.xlim([0, 256])
    plt.axis('off')
	#plt.savefig("test.png", bbox_inches='tight')

##Build probability function and show it based on beta function 
def probfunc(prior, enemyrange): #prior is rv function, enemyrange is np.linspace
    penemyrange = prior.pdf(enemyrange)
    plt.figure(figsize=(4,2))
    plt.plot(enemyrange, penemyrange)
    plt.title("PDF for Enemy Army Location")
    plt.xlabel("Distance from Enemy Base (px)")
    plt.ylabel("Probability of Army Presence")
    return prior

##Fit the probability function to a image plot as a function of distance from the known enemybase
def probplot(prior, enemybase, img):
    #make a grid to the size of the plot
    xgrid = np.linspace(0, 256, 256)
    ygrid = np.linspace(0, 256, 256)
    x, y = np.meshgrid(xgrid, ygrid)
    
    #create pdf from beta function as a function of distance from known enemybase reshaped by the x axis
    #ravel flattens out grid
    xygrid = np.c_[x.ravel(), y.ravel()]
    dist = spatial.distance.cdist(xygrid, [enemybase])
    probfind = prior.pdf(dist).reshape(x.shape)
    
    #put into probabilistic form
    pprior = np.exp(np.log(probfind))
    
    #make prob distribution sum = 1
    pprior = pprior / sum(pprior)
    
    #make color map
    probcolor = matplotlib.cm.get_cmap('RdYlGn')
    
    #take out zeros
    prior_plt = pprior.copy()
    prior_plt[prior_plt==0] = np.nan
    
    #Plot color map
    minimap(img)
    #draw contour map over z height values
    plt.contourf(x, y, prior_plt, cmap=probcolor, alpha = 0.5)
    return pprior, xygrid, x, y 

##Search an area with highest probabilty, then give an updated probability map
#coords = [25, 40]
def search(coords, pprior, xygrid, x, y, img):
    #Calculate spots of all distance from obs spot, based on scanner sweep radius of 13 px
    zdist = spatial.distance.cdist(xygrid, [coords])
    zdist2 = zdist.copy()
    
    #cut out any part that isn't within range of SC2 scanner sweep
    zdist2[zdist2<13] = 1
    zdist2[zdist2>13] = 0
    p_search = (.99*zdist2.reshape(x.shape))
    
    #remove any values below .05 to better visualize
    p_search_plot = p_search.copy()
    p_search_plot[p_search_plot<0.05] = np.nan
    
    #plot picture of search area
    probcolor = matplotlib.cm.get_cmap('RdYlGn')
    prior_plt = pprior.copy()
    prior_plt[prior_plt==0] = np.nan
    
    minimap(img)
    obs_cmap = matplotlib.cm.get_cmap('Blues')
    plt.contourf(x, y, prior_plt, cmap=probcolor, alpha = 0.5)
    plt.contourf(x, y, p_search_plot, cmap=obs_cmap, alpha = 0.6)
    
    #############TO DO : if search area finds army, end and mark success, if not mark fail
    #############if p_search_plot == armylocation then accuracylist = accuracylist+1, else +0
    
    #update prior to posterior using search probability data
    pposterior = np.exp(np.log(pprior) + np.log(1-p_search))
    #normalize to make probability map
    pposterior = pposterior / sum (pposterior)
    
    #remove 0s from map so colormap works properly
    pposterior_prob = pposterior.copy()
    pposterior_prob[pposterior_prob==0] = np.nan
    
    ##Show updated map adjusting probabilities based on the searched area
    minimap(img)
    #draw contour map over z height values, alpha is transparency
    plt.contourf(x, y, pposterior_prob, cmap=probcolor, alpha = 0.5)
    return pposterior
    #plt.savefig("test.png", bbox_inches='tight')


###Steps for a single use case
#1. Import a minimap
img = mpimg.imread('minimap2.png')
minimap(img)

#2. Calculate and plot a prior probability for that minimap
prior1 = scipy.stats.beta(.9, 2, loc = 0, scale = 256)
enemyrange1 = np.linspace(0, 256, 50)
prior = probfunc(prior1, enemyrange1)

#3. Show probability as a contour map across the plot
enemybase=[230,25]
pposterior, xygrid, x, y  = probplot(prior, enemybase, img)

#4. Show a search area based on highest probability of find, then update the posterior based on area searched, 
#4b. Can rerun this step to perform iterative search
newprior = pposterior.copy()
maxprob = np.argmax(newprior)
#np.multiply (x, x2), F continguous (column major style); why 1000 * np unravel index, unravel turns into a tuple of coordinate arrays
coords = (np.unravel_index(maxprob, newprior.shape, order = 'F'))
pposterior = search(coords, pposterior, xygrid, x, y, img)

