import json
from pymatgen import Lattice, Structure, Molecule, Composition
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
import numpy as np
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn
import pandas as pd
from itertools import chain
import os
import pprint
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.io.pwscf import PWInput, PWOutput

#Load data from folder called "data"
data_path = 'data/'

d_3d=loadfn(data_path + 'jdft_3d-5-23-2018.json',cls=MontyDecoder)
d=loadfn(data_path + 'jdft_2d-5-23-2018.json',cls=MontyDecoder)
#d=loadfn('data/jdft_2d.json',cls=MontyDecoder)
df_metals = pd.read_csv(data_path + 'metals_list.csv') #list of all metals on periodic table

#Miscellaneous Functions

#makes a list of perfect squares less than "number"
def perf_squares(number):
    perfect_squares_list = []
    if number >= 0:
        for i in range(1, int(number ** 0.5 + 1)):
            perfect_squares = i**2
            perfect_squares_list.append(perfect_squares)
    return perfect_squares_list

#checks if a space group has inversion symmetry
def has_inversion(group):
    ranges = ((2, 2), (10, 15), (47, 74), (83, 88), (123, 142), (147, 148), (162, 167), (175,176), (191, 194), (200, 206), (221, 230))
    nums = set(chain(*(range(start, end+1) for start, end in ranges)))
    return (group in nums)

#enables clickable urls in the dataframe
def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)

#Writes a poscar of a structure given the index "num" 
def poscar_wr(num,d):
    x= d[num]['final_str']
    fileout = Poscar(x)
    filename = d[num]['final_str'].formula.replace(" ", "") + '.vasp'
    fileout.write_file(filename)
    print("Wrote " + filename)
    
#Takes as input the dataframe and a list of elements (e.g. ['Mn','P','O'])
def search_for_element(df, el_list):
        searchfor = el_list
        df1 = df[df['Formula'].str.contains('|'.join(searchfor))]
        dfurl = df1.style.format({'Jarvis_URL': make_clickable, 'MP_URL': make_clickable})
        return dfurl
    
#Counts the number of metals per cell in a structure
def count_metals(d_i):
    comp = Composition(d_i['final_str'].formula.replace(" ", ""))
    comp_dict = comp.get_el_amt_dict()
    num_metals = 0.0
    for i in comp_dict:
        for index, row in df_metals.iterrows():
            if i == df_metals['Symbol'].iloc[index]:
                num_metals += comp_dict[i]
    return num_metals

#Gets Elastic Tensor in array form from json object
def get_et(elast_str=''):
    if elast_str == 'na':
        return 'na'
    else:
        cij=np.empty((6, 6), dtype=float)
        elast=np.array(elast_str.split(','),dtype='float')
        count=0
        for ii in range(6):
            for jj in range(6):
                cij[ii][jj]=elast[count]
                count=count+1
        et=ElasticTensor.from_voigt(cij)
        return et

#Calculates Universal Anisotropy Index from elastic tensor
def get_anisotropy(x):
    if x == 'na':
        return np.NaN
    else:
        try:
            return x.universal_anisotropy
        except:
            return np.NaN

#Returns a maximum in a list of Anisotropy indices for a given mpid
def get_anisotropy_list(dat_3d, mpid):
    anisotropy_list = []
    for i in dat_3d:
        if i['mpid'] == mpid:
            anisotropy_list.append(get_anisotropy(get_et(i['elastic'])))
            #pprint.pprint(i)
        x = np.array(anisotropy_list, dtype=np.float64)    
    return np.nanmax(x)

#Returns a maximum in a list of bulk OP band gaps for a given mpid
def get_bulk_gap_list(dat_3d, mpid):
    bulk_gap_list = []
    for i in dat_3d:
        if i['mpid'] == mpid:
            if i['op_gap'] == 'na':
                bulk_gap_list.append(np.NaN)
            else:
                bulk_gap_list.append(i['op_gap'])
        x = np.array(bulk_gap_list, dtype=np.float64)    
    return np.nanmax(x)

#returns kpoint grid size based on a 12x12x1 (21x21x1) grid for relaxation (Lumen SHG) of MoS2
def get_kpt_gridsize(d,num,relax=True,lumen=False):
    mos2_struct = d[255]['final_str']
    mos2_area = mos2_struct.volume/(mos2_struct.lattice.c)
    
    if relax != lumen: 
        if relax == True:
            kpt_den = mos2_area*144
        elif lumen == True:
            kpt_den = mos2_area*441
    else:
        return "Error"
            
    struct = d[num]['final_str']
    area = struct.volume/(struct.lattice.c)
    kpt_gridsize = 0
    for i, val in enumerate(perf_squares(2000)):
        if area*val < rel_kpt_den:
            kpt_gridsize = int(np.sqrt(val))
        else:
            kpt_gridsize = int(np.sqrt(val))
            break
    return kpt_gridsize

#Generate Quantum Espresso Input File
def generate_pw_input(num, d, calctype = 'vc-relax', path = 'QE_relax_inputs/ONCV/'):

    filname = d[num]['final_str'].formula.replace(" ", "")
    struct = d[num]['final_str']
    kpt_gridsize = get_kpt_gridsize(d,num,relax=True,lumen=False)
    kpts = [kpt_gridsize,kpt_gridsize,1]
    
    species_list = []
    for i in struct.species:
        if str(i) not in species_list:
            species_list.append(str(i))
    species_list
    pseudo = {}
    for el in species_list:
        pseudo[el] = el + '_ONCV_PBE_sr.upf'

    control ={'calculation': calctype,
        'restart_mode':'from_scratch',
        'prefix':'bn',
        'pseudo_dir' : '/home/beachk2/PSEUDO/upf_files/PBE/ONCVPSP-master/sg15/',
        'outdir': '/scratch/beachk2/Jarvis/IPA/' + filname,
        'wf_collect':True,
        'forc_conv_thr':1.0E-4,
        'verbosity':'high'
    }
    system = { 'ecutwfc' : 70,
        'occupations':'smearing',
        'smearing' : 'gaussian',
        'degauss':0.005,
        'force_symmorphic' : True
    }
    electrons = {    'mixing_mode' : 'plain',
        'mixing_beta' : 0.7,
        'conv_thr' :  1.0E-8
    }
    ions = {    'ion_dynamics' : 'bfgs'
    }
    if calctype == 'vc-relax':
        cell = {'cell_dofree' : '2Dxy'}
    kpoints_grid = kpts
    PWInput(struct,pseudo=pseudo,control=control,system=system,electrons=electrons,
            ions=ions, cell=cell, kpoints_grid =kpoints_grid).write_file(path +filname + ".relax.in")
    print("Wrote " + filname + ".relax.in")
    
    
#Building Dataset

keylist = ['Formula','OP_Gap', 'Bulk_OP_Gap', 'Bulk_Anisotropy', 'Atoms_per_cell','MBJ_Gap',
           'Mag_mom_per_cell', 'Mag_mom_per_metal', 'Mag_mom_per_area','Final_Energy_per_atom',
           'Exf_Energy_per_area','Space_Group','Has_Inversion','Jarvis_URL','MP_URL']
param_dict ={}
for i in keylist:
    param_dict[i] = {'ison':True, 'paramlist':[]}

#Omit columns from calculation here by setting "ison" to False. Default is True.
param_dict['MBJ_Gap']['ison'] = False
param_dict['MP_URL']['ison'] = True
param_dict['Mag_mom_per_cell']['ison'] = True
param_dict['Mag_mom_per_metal']['ison'] = False
param_dict['Mag_mom_per_area']['ison'] = True
param_dict['Bulk_Anisotropy']['ison'] = True
param_dict['Final_Energy_per_atom']['ison'] =False
param_dict['Atoms_per_cell']['ison'] =False
param_dict['Space_Group']['ison'] =False
param_dict['Bulk_OP_Gap']['ison'] =False    
    
for i in d:
    #useful parameters
    stoichiometry = i['final_str'].formula
    cell_vol = i['final_str'].volume
    area = cell_vol/i['final_str'].lattice.c
    num_atoms = i['final_str'].num_sites
    space_group = i['final_str'].get_space_group_info()[1]
    jarvis_url = str("https://www.ctcms.nist.gov/~knc6/jsmol/")+str(i['jid'])+str(".html")
    mp_url=str("https://materialsproject.org/materials/")+str(i['mpid'])+str('/#')
    magnetic_moment = i['magmom']['magmom_out']
    relaxed_energy = i['fin_en']
    exfoliation_energy = i['exfoliation_en']
    OptB88vdW_band_gap = i['op_gap']
    mBJ_band_gap = i['mbj_gap']
    if param_dict['Mag_mom_per_cell']['ison'] == True: 
        number_of_metals = count_metals(i)
        if number_of_metals != 0.0:
            magnetic_mom_metal = magnetic_moment/number_of_metals
        else:
            magnetic_mom_metal = 0.0
    else:
        magnetic_mom_metal = 'na'
    if param_dict['Bulk_Anisotropy']['ison'] == True:
        anisotropy = get_anisotropy_list(d_3d, i['mpid'])
    else:
        anisotropy = np.NaN
    if param_dict['Bulk_OP_Gap']['ison'] == True:
        bulk_gap = get_bulk_gap_list(d_3d, i['mpid'])
    else:
        bulk_gap = np.NaN
    
    #building lists
    param_dict['Formula']['paramlist'].append(stoichiometry)
    param_dict['Atoms_per_cell']['paramlist'].append(num_atoms)
    param_dict['OP_Gap']['paramlist'].append(OptB88vdW_band_gap)
    param_dict['MBJ_Gap']['paramlist'].append(mBJ_band_gap)
    param_dict['Mag_mom_per_cell']['paramlist'].append(magnetic_moment)
    param_dict['Mag_mom_per_area']['paramlist'].append(magnetic_moment/area) 
    param_dict['Mag_mom_per_metal']['paramlist'].append(magnetic_mom_metal)
    param_dict['Final_Energy_per_atom']['paramlist'].append(relaxed_energy/num_atoms)
    param_dict['Exf_Energy_per_area']['paramlist'].append(exfoliation_energy/area)
    param_dict['Space_Group']['paramlist'].append(space_group)
    param_dict['Has_Inversion']['paramlist'].append(has_inversion(space_group))
    param_dict['Jarvis_URL']['paramlist'].append(jarvis_url)
    param_dict['MP_URL']['paramlist'].append(mp_url)
    param_dict['Bulk_Anisotropy']['paramlist'].append(anisotropy)
    param_dict['Bulk_OP_Gap']['paramlist'].append(bulk_gap)
    
    
#Generating dataframe

param_dict['MBJ_Gap']['ison'] = False
param_dict['MP_URL']['ison'] = True
param_dict['Mag_mom_per_cell']['ison'] = True
param_dict['Mag_mom_per_metal']['ison'] = False
param_dict['Mag_mom_per_area']['ison'] = True
param_dict['Bulk_Anisotropy']['ison'] = True
param_dict['Final_Energy_per_atom']['ison'] =False
param_dict['Atoms_per_cell']['ison'] =False
param_dict['Space_Group']['ison'] =True
param_dict['Bulk_OP_Gap']['ison'] =False    

headers =[]
list_of_lists = []
for i in param_dict:
     if param_dict[i]['ison'] == True:
        headers.append(i)
        list_of_lists.append(param_dict[i]['paramlist'])

df = pd.DataFrame(list_of_lists)
df = df.transpose()
df.columns = headers
df_unfiltered = df.copy()

df = df[~df['OP_Gap'].isin(['na'])] #remove items with no band gap
df = df[df['OP_Gap']>.9] #Only include items with band gap > 1eV
df = df[~df['Has_Inversion'].isin([True])] #remove items with inversion symmetry


#Sort by desired parameter
#df = df.sort_values('OP_Gap',ascending=True)
#df = df.sort_values('Exf_Energy_per_area',ascending=True)
#df = df.sort_values('Final_Energy_per_atom',ascending=True)
#df = df.reindex(df.Mag_mom_per_area.abs().sort_values(ascending=False).index)
df = df.reindex(df.Bulk_Anisotropy.abs().sort_values(ascending=False).index)  #sort by absolute value of Anisotropy
#df = df.sort_values('Mag_mom_per_metal',ascending=False)
#df = df.sort_values('Space_Group',ascending=True)


#create dataframe with clickable urls
if param_dict['MP_URL']['ison'] == True:
    dfurl = df.style.format({'Jarvis_URL': make_clickable, 'MP_URL': make_clickable})
    
print("Number of Materials:",df.shape[0])