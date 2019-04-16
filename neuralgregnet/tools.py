import numpy as np 
from matplotlib import colors
import pickle
import math
import csv
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

genes = ["Kni","Hb","Gt","Kr"]
activations ={
    "sigmoid":keras.activations.sigmoid,
    "linear":keras.activations.linear
}
def create_model(description):
    model = keras.Sequential([keras.layers.Flatten(input_shape=(4,))]+[
            keras.layers.Dense(ss,activation=activations[tt]) for ss,tt in description
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),loss="mse")
    return model

def extract_time_series(data,mint=None,maxt=None,minc=None,maxc=None):
    """
    Extract the times series as numpy array for each of the genes:

    Args:
        data: pandas data frame
        mint: lower time index (if None, replaced by 0)
        maxt: upper time index (if None, replaced by len(time))
        minc: lower cell index (if None, replaced by 0)
        maxc: upper cell index (if None, replaced by len(cell))
    return:
        A dictionary. The first key is used for the selection of the mutant, the second for the gene.

    """
    data.sort_values("age")

    if not mint: mint=0
    if not maxt: maxt=None # Up to the end
    if not minc: minc=0
    if not maxc: maxc=None
    temp = list(data["dynamics"])
    Genes = list(temp[0].keys())
    time_series = {}
    times = np.array(list(data["age"]))[mint:maxt]
    for gg in Genes:
        time_series[gg] = np.array(np.array([val[gg] for val in temp]),dtype="float")[mint:maxt,minc:maxc]
    return times,time_series

def plot_data(positions,data,labels=None,mode="markers",title=""):
    """
    Plot the data set.
    
    Args:
        data (dict or ndarray): lines to plot. If the data are provided as ndarray, it is recommended to use the labels arguments.
        labels (list): Labels of the data's rows when data is provided as a ndarray
    """
    if type(data) is np.ndarray:
        if not labels:
            labels = ["G"+str(i) for i in range(len(data))]
        data = {key:dat for key,dat in zip(labels,data)}
    plotdata = []
    for key,dat in data.items():
        trace = go.Scatter(
            x = positions,
            y = dat,
            name=key,
            mode=mode
        )
        plotdata.append(trace)
    layout = go.Layout(
              title=title,
              yaxis = dict(title = "Quatity"),
              xaxis = dict(title = "AP position (% from anterior)"),
            
             )
    fig= go.Figure(data=plotdata, layout=layout)
    py.iplot(fig)

def to_numpy(data,genes):
    return np.array([data[gg] for gg in genes])

def apply_noise(data,level,tonumpy=True):
    new_data = {}
    for key,val in data.items():
        new_data[key] = np.array(val)+np.random.normal(0,scale=level,size=len(val))
        new_data[key][new_data[key]<0] = 0
    if tonumpy:
        new_data = to_numpy(new_data,genes)
    return new_data

def to_numpy(tab,genes = ["Gt","Kni","Kr","Hb"],normalize=False,nan=True):
    """
    Convert a pandas dataframe of the raw data into a numpy matrix

    Args:
        tab: pandas array of the raw data.
        genes: list of gene names. The function returns the gene columns in the same order as this list.
        normalize: Set range of variation of each gene to [0,1]
    
    """
    gene_data = np.float64([np.float64([np.float64(tab.loc[i][[gg]][0]) for gg in genes]) for i in tab.index])
    positions = np.linspace(0,100,gene_data.shape[-1])
    if normalize:
        for g in range(len(genes)):
            gene_data[:,g,:] = (gene_data[:,g,:].T-np.nanmin(gene_data[:,g,:],axis=1)).T
            gene_data[:,g,:] /= np.nanmax(gene_data[:,g,:]) 
    if not nan:
        gene_data = np.swapaxes(gene_data,0,2)
        select = np.array([not np.any(np.isnan(vec)) for vec in gene_data])
        gene_data = gene_data[select]
        positions = positions[select]
        gene_data = np.swapaxes(gene_data,0,2)
    age = np.float64(tab.age)
    return age,gene_data,genes,positions

activation_functions = {
    "sigmoid":lambda x:1/(1+np.exp(-x)),
    "linear":lambda x:x
}

class PseudoNet:
    def __init__(self,coef,intercept,activation):
        self.weights = [np.array(cc).T for cc in coef]
        self.bias = [np.array(ii) for ii in intercept]
        self.activations = [activation_functions[func] for func in activation]
        self.layers = []
    def predict(self,data,layer=None):
        if data.ndim==2:
            return np.apply_along_axis(lambda xx: self.predict(xx,layer=layer),1,data)
        result = np.array(data)
        self.layers = [result]
        ll = 0
        if not layer:
            layer = len(self.weights)+1
        for ww,bb,aa in zip(self.weights,self.bias,self.activations):
            preactivation = np.dot(ww,result) + bb
            result = aa(preactivation)
            self.layers.append(result)
            ll+=1
            if ll==layer:
                break
        return result
    
def split_data(data,ratios,shuffle=True):
    data = np.array(data)
    
    if shuffle:
        np.random.shuffle(data)
    ratios = np.int64(np.round(np.cumsum(np.array(ratios)/np.sum(ratios))*len(data)))
    spdata = []
    start = 0
    for end in ratios:
        spdata.append(data[start:end])
        start = end
    return spdata

def get_mapper(color_list,norm):
    colmap = colors.LinearSegmentedColormap.from_list(name="mymap",colors=color_list)
    mapper = lambda xx,norm=norm,colomap=colmap:colors.to_hex(colmap(int(255*(xx+norm)/(2*norm))))
    return mapper

def get_layer_output(model,X,layer_index):
    """
    Computes the output of one of the core layers
    
    Args:
        model : Model containing the desired layers
        X : Input
        layer_index: layer index or list of layer indexes
    """
    if type(layer_index) in  [list,range]:
        return [get_layer_output(model,X,ll) for ll in layer_index]
    else:
        model_bis = keras.Sequential(model.layers[:(layer_index+1)])
        return model_bis.predict(X)
    


def plot_model(model,X,color_list=['#fe5252', '#ffffff', '#46ca4c']):
    import plotly.offline as py
    py.init_notebook_mode()
    import plotly.graph_objs as go
    param = {"weights":[],"bias":[]}
    for layer in model.layers[1:]:
        pp = layer.get_weights()
        param["weights"].append(pp[0])
        param["bias"].append(pp[1])
    neuron_values = get_layer_output(model,X,range(len(model.layers)))
    neuron_values = [nn[0] for nn in neuron_values]
  
    max_intens = np.max([np.max(val) for val in neuron_values])
    colormapperN = get_mapper(color_list,max_intens)
    max_intens = np.max([np.max(np.abs(ww)) for ww in param["weights"]])
    colormapperE = get_mapper(color_list,max_intens)
    
    edge_traces = [] 
    scalex = 1
    scaley = 1
    radius = 1
    spacex = 2
    spacey = 1
    Nmax = np.max([len(layer) for layer in neuron_values])
    
    Hmax = Nmax*(radius+spacey) - spacey
    posx=posy=parent_x=parent_y=None

    xx = []
    yy = []
    colors = []
    infos = []

    for i,lay in enumerate(neuron_values):
        h = len(lay)*(radius+spacey) - spacey
        shifty = (Hmax-h)/2
        if posx:
            parent_x = posx
        posx = 0.5 + i*(radius+spacex)
        temp_parent_y = []        
        for j,val in enumerate(lay):
            posy = 0.5 + (len(lay)-j-1)*(radius+spacey) + shifty
            xx.append(posx)
            yy.append(posy)            
            temp_parent_y.append(posy)                        
            colors.append(colormapperN(val))
            info_node = str(val)
            if i>0:     
                for k in range(len(neuron_values[i-1])):                  
                    x0,y0 = parent_x,parent_y[k]                        
                    x1,y1 = posx,posy
                    val = param["weights"][i-1][k,j]
                    col = colormapperE(val)
                    edge = go.Scatter(x=np.linspace(x0,x1,10),y=np.linspace(y0,y1,10),text=[""]+["{}".format(val)]*8+[""],mode="lines",line = dict(color = col,width = 4),marker=go.scatter.Marker(opacity=0),hoverinfo='text')
                    edge_traces.append(edge)
                info_node = "{}\n(b={})".format(info_node,param["bias"][i-1][j])
            infos.append(info_node)
        parent_y=temp_parent_y
        ldict = dict(autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False)
        layout = go.Layout(xaxis=ldict,yaxis=ldict,hovermode = 'closest',showlegend=False)
    node_trace = go.Scatter(x=xx,y=yy,text=infos,mode="markers",marker=go.scatter.Marker(color=colors,size=50,line=dict(width=2,color="grey")),hoverinfo='text')
    return edge_traces+[node_trace],layout
    
def load_data_set(name):
    with open(name,"rb") as tfile:
        data = pickle.load(tfile)["data"]
    data = np.swapaxes(data,1,2)
    shape = data.shape
    data = data.reshape(shape[0]*shape[1],shape[2])
    return data

def h1VSh2( autoencoder,inputs,age, cut, off,toFile=False,file='hiddenNodes/hiddenNodes1.csv' ):
    get_middle_layer_output = K.function([autoencoder.layers[0].input],[autoencoder.layers[1].output])
    h1s=[]
    h2s=[]
    if toFile:
        f = open(file, 'w')
    for i in range(age):
                print(i)
                h1=[]
                h2=[]
                #loop trough positions
                for j in range(cut,off):
                    inp=inputs[i,:,j].reshape((1,4))
                    layer_output = get_middle_layer_output([inp])[0][0]
                    h1.append(layer_output[0])
                    h2.append(layer_output[1])
                
                if toFile:
                    writer = csv.writer(f)#, delimiter='\t')
                    writer.writerows(zip(h1,h2))
                
                plt.plot(h1,h2, '.', markersize=1)
                plt.xlabel("h1")
                plt.ylabel("h2")
                h1s.append(h1)
                h2s.append(h2)
    if toFile:
        f.close()
    h1s=np.array(h1s)
    h2s=np.array(h2s)
    return h1s,h2s
 
 #input: 2 hidden nodes (a,b), hiddenToPosition is the map shape=(lenght of positions, 2) the 2 hiddens nodes in order of position and position is the positions equivalent to hiddenToPosition
 #output: position prediction
def hiddenNodesToPosition(a,b,hiddenToPosition,positions):
    shortest=math.sqrt((a-hiddenToPosition[0][0])**2 + (b-hiddenToPosition[0][1])**2)
    index=0
    for i in range(1,hiddenToPosition.shape[0]):
        point=hiddenToPosition[i]
        distance= math.sqrt((a-point[0])**2 + (b-point[1])**2)
        if distance < shortest:
            shortest=distance
            index=i
    pos= positions[index]        
    return pos,index,shortest
