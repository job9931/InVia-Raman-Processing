import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import peakutils
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths




'Define Functions'
'--------------------------------------------------'
def reduce_data(data):
    'Takes 2D array and returns 3D array for energy and 3D array for intensity'

    'Get unique x and y'
    x = np.unique(data[:,0])
    y = np.unique(data[:,1])
    
    flag = True
    for ix in range(len(x)):
        boolx = np.where(data[:,0]==x[ix])
        subdata = data[boolx]

        for iy in range(len(y)):
            booly = np.where(subdata[:,1]==y[iy])
            subsubdata = subdata[booly]
            
            if flag: #Use if statement so this block is only excuted once. 
                E = np.zeros((len(x), len(y), len(subsubdata)))
                I = np.zeros((len(x), len(y), len(subsubdata)))
                flag = False
            
            E[ix][iy] = subsubdata[:,2]
            I[ix][iy] = subsubdata[:,3]


    return x, y, E, I


def peak_detection(E,I,height=300,xpoints=10,ypoints=10,baseline_order=4):

    'Numpy arrays to store peak prominance and positions'
    peak_intensity = np.zeros((xpoints,ypoints))
    peak_position = np.zeros((xpoints,ypoints))
    peak_width = np.zeros((xpoints,ypoints))

    for ix in range(xpoints):
        for iy in range(ypoints):
            x = E[ix,iy,:]
            y = I[ix,iy,:]
            baseline = peakutils.baseline(y,baseline_order)
            y_adjusted = y - baseline
            peaks,_ = find_peaks(y_adjusted,height)
            if peaks.size == 0:
                peak_intensity[ix][iy]=np.nan
                peak_position[ix][iy] = np.nan
                peak_width[ix][iy] = np.nan
            else:
                peak_heights = y_adjusted[peaks]
                peak_positions = x[peaks]
                max_peak_height = np.amax(peak_heights)
                max_peak_index  = np.where(y_adjusted==max_peak_height)
                max_peak_pos = x[max_peak_index]
                max_peak_wid = peak_widths(y_adjusted,max_peak_index[0])
                
                peak_intensity[ix][iy]=max_peak_height
                peak_position[ix][iy]=max_peak_pos
                peak_width[ix][iy] = max_peak_wid[0]

    
    return peak_intensity, peak_position, peak_width


def plot_maps(map_list,cbar_label_list,map_colour='jet',null_colour='dimgrey',extent = (0,10,0,10)):

    fig,ax=plt.subplots(1,len(map_list),sharey='all')
    for i in range(len(map_list)):

        cmap = cm.get_cmap(map_colour)
        cmap.set_bad(color=null_colour)
        im = ax[i].imshow(map_list[i],cmap=cmap,extent=extent,interpolation='nearest',aspect='equal')
        cbar = fig.colorbar(im,ax=ax[i], orientation='horizontal')
        cbar.set_label(cbar_label_list[i])
        ax[i].set_xlabel('$x\,(\mu m)$')

    ax[0].set_ylabel('$y\,(\mu m)$')

    return fig

def plot_histograms(map_list,yaxlabel,xaxlabel,fc_list,ec_list,bin_num=5,sharex=False,sharey='row'):

    fig,ax = plt.subplots(1,len(map_list),sharex = sharex, sharey=sharey)
    for i in range(len(map_list)):
        flat_map = map_list[i].flatten()
        flat_map = flat_map[np.where(~np.isnan(flat_map))]
        ax[i].hist(flat_map,bins=bin_num,color=fc_list[i],ec=ec_list[i])
        ax[i].set_xlabel(xaxlabel[i])
        if sharey =='row':
            ax[0].set_ylabel(yaxlabel[0])
        else:
            ax[i].set_ylabel(yaxlabel[i])
    plt.tight_layout()

    return fig

def heatmap(map_list,cbar_label='I$_{2D}/I$_G$',map_colour='jet',null_colour='dimgrey',extent = (0,10,0,10)):

    for i in range(len(map_list)):
        fig=plt.figure()
        cmap = cm.get_cmap(map_colour)
        cmap.set_bad(color=null_colour)
        im = plt.imshow(map_list[i],cmap=cmap,extent=extent,interpolation='nearest',aspect='equal')
        cbar = fig.colorbar(im, orientation='horizontal')
        cbar.set_label(cbar_label)
        plt.set_xlabel('$x\,(\mu m)$')
        plt.set_ylabel('$y\,(\mu m)$')
        plt.savefig(Path.home()/'Raman'/'Ch'+str(i+1)/'Figures'/'Ch'+str(i+1)+'_r_map.png',dpi=300)
        plt.savefig(Path.home()/'Raman'/'Ch'+str(i+1)/'Figures'/'Ch'+str(i+1)+'_r_map.eps')
        plt.show()
        plt.close()


def remove_nan(array):
    return array[np.where(~np.isnan(array))]

def save_map_stats(raman_map):
    raman_map = remove_nan(raman_map)

    

'--------------------------------------------------'




'Import data'
'--------------------------------------------------'
path_2d  = Path.home()/'Programs'/'2d_maps'
path_g = Path.home()/'Programs'/'g_maps'

files_2d = sorted(list(path_2d.glob('*.txt')),key=lambda path: int(path.name.split('_')[0]))
files_g = sorted(list(path_g.glob('*.txt')),key=lambda path: int(path.name.split('_')[0]))

data_2d = [np.genfromtxt(file) for file in files_2d]
data_g = [np.genfromtxt(file) for file in files_g]

cbar_label='E$_G\,$(cm$^{-1})$'
map_colour='jet'
null_colour='dimgrey'
extent = (0,10,0,10)



'Create arrays to store statistics'
mean_intensities = np.zeros((3,len(data_2d)))
std_intensities = np.zeros((3,len(data_2d)))

mean_pos = np.zeros((2,len(data_2d)))
std_pos = np.zeros((2,len(data_2d)))

mean_width = np.zeros((2,len(data_2d)))
std_width = np.zeros((2,len(data_2d)))

for i in range(len(data_2d)):
    x2d,y2d,E2d,I2d = reduce_data(data_2d[i])
    xg,yg,Eg,Ig = reduce_data(data_g[i]) 
    I_map2d,E_map2d,W_2d = peak_detection(E2d,I2d)
    I_mapG,E_mapG,W_g = peak_detection(Eg,Ig)
    Ig_ratio= np.divide(I_map2d,I_mapG)

    bool_EG = np.where(E_mapG>1650)
    E_mapG[bool_EG] = np.nan
    bool_EG = np.where(E_mapG<1410)
    E_mapG[bool_EG] = np.nan

    #fig=plt.figure()
    #cmap = cm.get_cmap(map_colour)
    #cmap.set_bad(color=null_colour)
    #im = plt.imshow(E_mapG,cmap=cmap,extent=extent,interpolation='nearest',aspect='equal')
    #cbar = fig.colorbar(im)
    #cbar.set_label(cbar_label,fontsize=18)
    #plt.xlabel('$x\,(\mu m)$',fontsize=18)
    #plt.ylabel('$y\,(\mu m)$',fontsize=18)
    #plt.show()
    #np.savetxt(str(i)+'Imap2d.txt',I_map2d)
    #np.savetxt(str(i)+'ImapG.txt',I_mapG)
    #np.savetxt(str(i)+'Emap2d.txt',E_map2d)
    #np.savetxt(str(i)+'EmapG.txt',E_mapG)
    #np.savetxt(str(i)+'Wmap2d.txt',W_2d)
    #np.savetxt(str(i)+'WmapG.txt',W_g)


    'Intensity stats'
    I_map2d = remove_nan(I_map2d)
    I_mapG = remove_nan(I_mapG)
    Ig_ratio = remove_nan(Ig_ratio)

    E_map2d = remove_nan(E_map2d)
    E_mapG = remove_nan(E_mapG)

    W_2d = remove_nan(W_2d)
    W_g = remove_nan(W_g)
    

    meanI2 = np.mean(I_map2d)
    meanIG = np.mean(I_mapG)
    meanIR = np.mean(Ig_ratio)
    stdI2 = np.std(I_map2d)
    stdIG = np.std(I_mapG)
    stdIR = np.std(Ig_ratio)
    
    mean_intensities[0,i] = meanI2
    mean_intensities[1,i] = meanIG
    mean_intensities[2,i] = meanIR

    std_intensities[0,i] = stdI2
    std_intensities[1,i] = stdIG
    std_intensities[2,i] = stdIR

    'Position stats'
    meanE2 = np.mean(E_map2d)
    meanEg = np.mean(E_mapG)
    stdE2 = np.std(E_map2d)
    stdEg = np.std(E_mapG)

    mean_pos[0,i] = meanE2
    mean_pos[1,i] = meanEg
    std_pos[0,i] = stdE2
    std_pos[1,i] = stdEg

    'Width stats'
    meanW2 = np.mean(W_2d)
    meanWg = np.mean(W_g)
    stdW2 = np.std(W_2d)
    stdWg = np.std(W_g)

    mean_width[0,i] = meanW2
    mean_width[1,i] = meanWg
    std_width[0,i] = stdW2
    std_width[1,i] = stdWg


np.savetxt('mean_intensities.txt',mean_intensities)
np.savetxt('std_intensities.txt',std_intensities)
np.savetxt('mean_positions.txt',mean_pos)
np.savetxt('std_positions.txt',std_pos)
np.savetxt('mean_width.txt',mean_width)
np.savetxt('std_widths.txt',std_width)
exit()


'Reduce data'
'--------------------------------------------------'
x2d,y2d,E2d,I2d = reduce_data(data_2d)
xg,yg,Eg,Ig = reduce_data(data_g)              


'Calculate properties of 2D and G peaks'
'--------------------------------------------------'
I_map2d,E_map2d,W_2d = peak_detection(E2d,I2d)
I_mapG,E_mapG,W_g = peak_detection(Eg,Ig)
Ig_ratio= np.divide(I_map2d,I_mapG)


'Create peak intensity heatmaps'
'--------------------------------------------------'
map_listI=[1e-3*I_map2d,I_mapG,Ig_ratio]
cbar_labelsI = ['I$_{2D}\,(10^3\,$a.u.)','I$_G\,$(a.u.)','I$_{2D}$/I$_G\,$(a.u.)']
Ifig = plot_maps(map_listI,cbar_labelsI,map_colour='jet')
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_I_map.png',dpi=300)
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_I_map.eps')
plt.show()
plt.close()


'Create peak position heatmaps'
'--------------------------------------------------'
bool_E2d = np.where(E_map2d>2900)
E_map2d[bool_E2d] = np.nan
bool_EG = np.where(E_mapG>1700)
E_mapG[bool_EG] = np.nan
bool_EG = np.where(E_mapG<1410)
E_mapG[bool_EG] = np.nan
map_listE = [E_map2d,E_mapG]
cbar_labelsE=['$E_{2D}\,$(cm$^{-1}$)','$E_G\,$(cm$^{-1}$)']
Efig = plot_maps(map_listE,cbar_labelsE,map_colour='jet')
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_E_map.png',dpi=300)
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_E_map.eps')
plt.show()
plt.close()


'Create peak width heatmaps'
'--------------------------------------------------'
#bool_W2d = np.where(W_2d>40)
#W_2d[bool_W2d] = np.nan
bool_WG = np.where(W_g>40)
W_g[bool_WG] = np.nan
map_listW = [W_2d,W_g]
cbar_labelsW=['$\Gamma_{2D}\,$(cm$^{-1}$)','$\Gamma_G\,$(cm$^{-1}$)']
Wfig = plot_maps(map_listW,cbar_labelsW,map_colour='jet')
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_W_map.png',dpi=300)
plt.savefig(Path.home()/'Raman'/'Ch16'/'Figures'/'Ch16_W_map.eps')
plt.show()
plt.close()



'Create intensity histograms'
'--------------------------------------------------'
map_list = [E_map2d,E_mapG]
xaxlabel = ['I$_{2D}\,$(cm$^{-1}$)','I$_G\,$(cm$^{-1}$)','I$_{2D}$/I$_G\,$(cm$^{-1}$']
fc_list = ['lightblue','moccasin','lightgreen']
ec_list = ['midnightblue','darkorange','darkgreen']
fig1 = plot_histograms([I_map2d,I_mapG,Ig_ratio],['Count'],xaxlabel,fc_list,ec_list,bin_num=7)
#plt.savefig(Path.home()/'Raman'/'Ch1_I_hist.png',dpi=300)
#plt.savefig(Path.home()/'Raman'/'Ch1_I_hist.eps')
plt.show()
plt.close()


'Create position histograms'
'--------------------------------------------------'
map_list = [E_map2d,E_mapG]
xaxlabel = ['E$_{2D}\,$(cm$^{-1}$)','E$_G\,$(cm$^{-1}$)']
fc_list = ['lightblue','moccasin']
ec_list = ['midnightblue','darkorange']
fig1 = plot_histograms(map_list,['Count'],xaxlabel,fc_list,ec_list,bin_num=7)
#plt.savefig(Path.home()/'Raman'/'Ch1_E_hist.png',dpi=300)
#plt.savefig(Path.home()/'Raman'/'Ch1_E_hist.eps')
plt.show()
plt.close()


'Create position histograms'
'--------------------------------------------------'
map_list = [W_2d,W_g]
xaxlabel = ['$\Gamma_{2D}\,$(cm$^{-1}$)','$\Gamma_G\,$(cm$^{-1}$)']
fc_list = ['lightblue','moccasin']
ec_list = ['midnightblue','darkorange']
fig1 = plot_histograms(map_list,['Count'],xaxlabel,fc_list,ec_list,bin_num=7)
#plt.savefig(Path.home()/'Raman'/'Ch1_W_hist.png',dpi=300)
#plt.savefig(Path.home()/'Raman'/'Ch1_W_hist.eps')
plt.show()
plt.close()


'Save processed maps (intensity, position, width)'
'--------------------------------------------------'
savepath_I2d = Path.home()/'Raman'/'Ch16_I2d.txt'
savepath_E2d = Path.home()/'Raman'/'Ch16_E2d.txt'
savepath_W2d = Path.home()/'Raman'/'Ch16_W2d.txt'

savepath_IG =  Path.home()/'Raman'/'Ch16_IG.txt'
savepath_EG =  Path.home()/'Raman'/'Ch16_EG.txt'
savepath_WG = Path.home()/'Raman'/'Ch16_WG.txt'

savepath_I_ratio =  Path.home()/'Raman'/'Ch16_I_ratio.txt'

np.savetxt(savepath_I2d,I_map2d)
np.savetxt(savepath_E2d,E_map2d)
np.savetxt(savepath_IG,I_mapG)
np.savetxt(savepath_EG,E_mapG)
np.savetxt(savepath_I_ratio,Ig_ratio)
np.savetxt(savepath_W2d,W_2d)
np.savetxt(savepath_WG,W_g)


exit