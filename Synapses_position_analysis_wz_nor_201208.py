import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import glob
import os

class Batch_Read():
    def __init__(self, data_path, ref_suffix=None, data_suffix=None):
        self.data_path = glob.glob(data_path)[0]
        self.file_list = os.listdir(self.data_path)
        self.ref_suffix = ref_suffix
        self.data_suffix = data_suffix
        
    def pair_data_ref(self):
        ref_suffix = self.ref_suffix
        file_list = self.file_list
        data = []
        ref = []
        if ref_suffix is not None:
            ref_len = len(ref_suffix)
            for item in file_list:
                if item[-len(ref_suffix)-4:-4] == ref_suffix:
                    ref.append(item)  
                elif ref_suffix not in item:
                    data.append(item)
        elif ref_suffix is None:
            ref_len = 0
            for item in file_list:
                if item[-7:-4] == 'ref':
                    ref.append(item)  
                elif 'ref' not in item:
                    data.append(item)
        if len(ref) != len(data):
            return 'data and ref number not matched'

        data_ref = []
        for i in range(len(data)):
            if ref_len == 0:
                ref_prefix = ref[i][:-8]
            else:
                ref_prefix = ref[i][:-ref_len-5]

            if ref_prefix in data[i]:
                data_ref.append((data[i], ref[i]))
        if len(data_ref) == len(data):
            return data_ref
        else:
            return 'data and ref name not matched'

    def rename_data_ref(self, data_suffix=None, new_data_suffix=None, new_ref_suffix=None, ref_suffix=None, change_name=False):
        file_list = self.file_list
        if ref_suffix is None:
            ref_suffix = self.ref_suffix
        if data_suffix is None:
            data_suffix = self.data_suffix
        if ref_suffix is None:
            return 'Please specify ref suffix'
        if data_suffix is None:
            return 'Please specify data suffix'
        
        old_data_name = []
        old_ref_name = []
        for item in file_list:
            if ref_suffix in item:
                old_ref_name.append(item)
            elif data_suffix in item:
                old_data_name.append(item)
        if len(old_ref_name) != len(old_data_name):
            return 'data and ref number not matched'
        
        old_new_data_name=[]
        old_new_ref_name=[]
        for i in range(len(old_data_name)):
            data_name_root = old_data_name[i].replace(data_suffix, '')
            ref_name_root = old_ref_name[i].replace(ref_suffix, '')
            if data_name_root != ref_name_root:
                return f'{i} {data_name_root} and {ref_name_root} do not match'
            new_data_name = old_data_name[i].replace(data_suffix, new_data_suffix)
            new_ref_name = old_ref_name[i].replace(ref_suffix, new_ref_suffix)
            old_new_data_name.append((old_data_name[i], new_data_name))
            old_new_ref_name.append((old_ref_name[i], new_ref_name))

        if change_name is True:
            for old_new in old_new_data_name:
                os.rename(f'{self.data_path}\\{old_new[0]}',f'{self.data_path}\\{old_new[1]}')
            for old_new in old_new_ref_name:
                os.rename(f'{self.data_path}\\{old_new[0]}',f'{self.data_path}\\{old_new[1]}')
                
        return old_new_data_name, old_new_ref_name
        
class Syn_Position(): #position data exported from imaris
    #ref file should contain at least 3 points, 
    #the first one is the original point,
    #the second one defines the midline,
    #the third one determines the side of interest
    #normalization is based on those three ref points

    def __init__(self, data_path, filename, refname=None, ref_suffix=None, nor_x=-1675, nor_y=1657):
        self.data_path = data_path
        self.filename = filename
        self.refname = refname
        self.ref_suffix = ref_suffix
        
        # for PCRt default 1657, 1st origion, 2nd 4V, 3rd lateral 7N
        #for LPGi, nor_x=-1160, nor_y=1600, 3rd amb
        #for MdV, nor_x=-1218, nor_y=1525, 3rd amb
        self.nor_x = nor_x
        self.nor_y = nor_y
        
    def read_raw_data(self):
        data_filepath = glob.glob(self.data_path +'\\'+ '*' + self.filename)
        filename_suffix = self.filename[-3:]
        if filename_suffix == 'xls':
            data = pd.read_excel(data_filepath[0], header=None, names=['y','x'], skiprows=2, usecols=[0,1])
        elif filename_suffix == 'csv':
            data = pd.read_csv(data_filepath[0], header=None, names=['y','x'], skiprows=4, usecols=[0,1])
        else:
            return 'Only read xls or csv file'
        return data
    
    def read_raw_ref(self, ref_suffix=None):
        ref_filepath = glob.glob(self.data_path +'\\'+ '*' + self.refname)
        if ref_suffix is None:
            ref_suffix = self.ref_suffix
        if self.ref_suffix is None:
            ref_suffix = self.refname[-7:]
        if 'ref' not in ref_suffix:
            print('check if ref is correct')
        if '.xls' in self.refname:
            ref = pd.read_excel(ref_filepath[0], header=None, names=['y','x'], skiprows=2, usecols=[0,1])
        elif '.csv' in self.refname:
            ref = pd.read_csv(ref_filepath[0], header=None, names=['y','x'], skiprows=4, usecols=[0,1])
        else:
            return 'Only read xls or csv ref'
        return ref
    
    def read_nor_data(self):
        data_filepath = glob.glob(self.data_path +'\\'+ '*' + self.filename)
        filename_suffix = self.filename[-7:]
        if filename_suffix == 'nor.xls':
            data = pd.read_excel(data_filepath[0], header=None, names=['x','y'], usecols=[0,1])
        elif filename_suffix == 'nor.csv':
            data = pd.read_csv(data_filepath[0], header=None, names=['x','y'], usecols=[0,1])
        else:
            return 'Only read xls or csv file'
        return data
    
    def read_nor_ref(self):
        ref_filepath = glob.glob(self.data_path +'\\'+ '*' + self.refname)
        refname_suffix = self.refname[-11:]
        if refname_suffix == 'ref_nor.xls':
            ref = pd.read_excel(ref_filepath[0], header=None, names=['x','y'], usecols=[0,1])
        elif refname_suffix == 'ref_nor.csv':
            ref = pd.read_csv(ref_filepath[0], header=None, names=['x','y'], usecols=[0,1])
        else:
            return 'Only read xls or csv ref'
        return ref
    
    def preprocess_data(self, exp=None, ref_exp=None):
        data = self.read_raw_data()
        ref = self.read_raw_ref()
        
        #shift the corordinate to origin
        data['x_ref'] = data['x'] - ref['x'][0]
        data['y_ref'] = data['y'] - ref['y'][0]
        ref['x_ref'] = ref['x'] - ref['x'][0]
        ref['y_ref'] = ref['y'] - ref['y'][0]
        
        # set an angle to rotate the points until the midline is vertical
        theta = np.arctan(ref['x_ref'][1]/ref['y_ref'][1])
        R = np.matrix([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
        
        # rotate the data and ref points according to the angle
        data_ref = np.matrix([data['x_ref'], data['y_ref']])
        ref_ref = np.matrix([ref['x_ref'], ref['y_ref']])
        data_nor = R @ data_ref
        ref_nor = R @ ref_ref
        data_nor = data_nor.T
        ref_nor = ref_nor.T
        
        #flip the data and ref points
        if ref_nor[1,1]<0:
            data_nor[:,1] = -data_nor[:,1]
            ref_nor[:,1] = -ref_nor[:,1]

        if ref_nor[2,0]>0:
            ref_nor[:,0] = -ref_nor[:,0]
            data_nor[:,0] = -data_nor[:,0]
            
        #normalize the points to atlas
        noridx_y = self.nor_y/ref_nor[1, 1]
        noridx_x = self.nor_x/ref_nor[2, 0]
        data_nor[:,1] = noridx_y * data_nor[:,1]
        ref_nor[:,1] = noridx_y * ref_nor[:,1]
        ref_nor[:,0] = noridx_x * ref_nor[:,0]
        data_nor[:,0] = noridx_x * data_nor[:,0]
            
        # export normalized data
        data_pp = pd.DataFrame(data_nor, columns=['x','y'])
        ref_pp = pd.DataFrame(ref_nor, columns=['x','y'])
        
        if exp == True:
            if self.refname[-7:-4] == 'ref':
                refname_pp = f"{self.refname[0:-4]}_nor.csv"
            else:
                refname_pp = f"{self.refname[0:-4]}_ref_nor.csv"
            data_pp.to_csv(f"{self.filename[0:-4]}_nor.csv", index=False)
            ref_pp.to_csv(refname_pp, index=False)
            
        if ref_exp == True:
            return ref_pp
        else:
            return data_pp
            
    def split_data(self, one_side=None, contra_side=None, delim=None, left=None, right=None):
        if self.filename[-7:-4] == 'nor':
            f_data = self.read_nor_data()
        else:
            f_data = self.preprocess_data()
            
        if one_side is True:
            if contra_side is True:
                data_one = f_data[(f_data.x>0)]
            else:
                data_one = f_data[(f_data.x<0)]
            if delim is not None:
                if type(delim) == np.float64:
                    delimiter = delim
                    if left is True:
                        data_sub = data_one[(data_one.x<delimiter)]
                        region = 'left_one_side'
                    elif right is True:
                        data_sub = data_one[(data_one.x>delimiter)]
                        region = 'right_one_side'
                    else:
                        raise Exception("Please set left or right")
                elif type(delim) == list:
                    x1, y1 = delim[0]
                    x2, y2 = delim[1]
                    if x1 == x2:
                        delimiter = x1                    
                        if left is True:
                            data_sub = data_one[(data_one.x<delimiter)]
                            region = 'left_one_side'
                        elif right is True:
                            data_sub = data_one[(data_one.x>delimiter)]
                            region = 'right_one_side'
                        else:
                            raise Exception("Please set left or right")                        
                    else:
                        k = (y1-y2)/(x1-x2)
                        if left is True:
                            data_sub = data_one[(data_one.y > (k*(data_one.x-x1)+y1))]
                            region = 'left_one_side'
                        elif right is True:
                            data_sub = data_one[(data_one.y < (k*(data_one.x-x1)+y1))]
                            region = 'right_one_side'
                        else:
                            raise Exception("Please set left or right")
                else:
                    raise Exception("delim should be float or list")
            else:
                data_sub = data_one
                region = 'one_side'
        else:
            data_sub = f_data
            region = 'both_sides'
            
        return data_sub, region
    
    def scatter_plot(self, xmin=-4.5, xmax=1, ymin=-2, ymax=3, size=2, exp=None, one_side=True, label=None, filled=None, delim=None, left=None, right=None):
        data_sub, region = self.split_data(one_side=one_side, delim=delim, left=left, right=right)
            
        x = data_sub['x']/1000
        y = data_sub['y']/1000
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.scatter(x, y, s=size, alpha=0.2,c='Purple')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.axis('equal')
        
        # Save plot
        if exp == True:
            plt.savefig(f'spot_{region}_{self.filename[:-4]}.svg')
            
    def contour_plot(self, xmin=-2.5, xmax=0.5, ymin=-1, ymax=2, line_num=12, exp=None, one_side=True, label=None, filled=None, delim=None, left=None, right=None):
        data_sub, region = self.split_data(one_side=one_side, delim=delim, left=left, right=right)

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f_syn = np.reshape(np.zeros(100**2), xx.shape)
            
        x = data_sub['x']/1000
        y = data_sub['y']/1000

        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f_syn += f

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # Contour plot
        cset = ax.contour(xx, yy, f_syn, line_num, colors='b')
        
        # Contourf plot
        if filled == True:
            cfset = ax.contourf(xx, yy, f_syn, cmap='Blues')
        ## Or kernel density estimate plot instead of the contourf plot
        #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        
        # Label plot
        if label == True:
            ax.clabel(cset, inline=1, fontsize=10)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.axis('equal')
        
        # Save plot
        if exp == True:
            plt.savefig(f'syn_{region}_{self.filename[:-4]}.svg')
    
    def kde1d_x_plot(self, xmin=-2.5, xmax=0.5, bandwidth=0.1, exp=None, one_side=True, delim=None, left=None, right=None):
        data_sub, region = self.split_data(one_side=one_side, delim=delim, left=left, right=right)
            
        x = data_sub['x']/1000
        
        x_d = np.linspace(xmin, xmax, 100)
        kde = KernelDensity(bandwidth, kernel='gaussian')
        kde.fit(x[:, None])
        logprob = kde.score_samples(x_d[:, None])
        plt.plot(x_d, np.exp(logprob))
        plt.ylim(-0.02, 1.4)
        
        # Save plot
        if exp == True:
            plt.savefig(f'kde1d_x_{region}_{self.filename[:-4]}.svg') 

    def kde1d_y_plot(self, ymin=-1, ymax=2, bandwidth=0.1, x_axis=True, y_axis=False, exp=None, one_side=True, contra_side=None, delim=None, left=None, right=None, yest_exp=None):
        data_sub, region = self.split_data(one_side=one_side, contra_side=contra_side, delim=delim, left=left, right=right)
            
        y = data_sub['y']/1000
        
        y_d = np.linspace(ymin, ymax, 100)
        kde = KernelDensity(bandwidth, kernel='gaussian')
        kde.fit(y[:, None])
        logprob = kde.score_samples(y_d[:, None])
        plt.plot(y_d, np.exp(logprob))
        plt.ylim(-0.02, 2.3)
        
        # Save plot
        if exp == True:
            plt.savefig(f'kde1d_y_{region}_{self.filename[:-4]}.svg') 
            
        if yest_exp is True:
            return pd.DataFrame({'y':y_d, 'est':np.exp(logprob)})