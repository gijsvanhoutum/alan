import numpy as np
import cv2

    
    
    
    
    
    
def getChannel(image):
    
    
    #R,G,B = cv2.split(image) 
    #im = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    #H,L,S = cv2.split(im) 
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    #L = (R+G+B)/3
    return gray

def matching( image, left, right):
    
    R,G,B = cv2.split(image)       
    channels = [R,G,B]        

    sses = [template(ch,left=left,right=right) for ch in channels]
    nsse = [normalize(sse,left=left,right=right) for sse in sses]
    total = euclidean(nsse)
    
    return total

def mergeIntervals(exclude):
    
    sorted_intervals = sorted(exclude, key=lambda x: x[0])
    interval_index = 0

    for  i in sorted_intervals:

        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]

    return sorted_intervals[:interval_index+1]
    
def clustering( image, left=1, right=-1 ):
    assert(left >= 0 and left < image.shape[1])
    assert(right >= 0 and right < image.shape[1])
    assert(left < right)
    
    error = matching(image,left,right) 

    verror = np.copy(error)
    ex_list = []
    comb_var = np.var(error)
    cond = None
    i=0

    while(i < 6):
        li,ri,am = getTriangle(verror,left,right)   

        verror[li:ri] = np.nan
        ex_list.append([li,ri])          
        #ex_list = mergeIntervals(ex_list)
        
        cond = getCondition(verror,ex_list)
        var1,var2,mn1,mn2 = getVariance(error,cond)
        

        """
        if var1+var2 > comb_var:
            break
        else:
            combvar = var1+var2
        """
        i +=1
        
    return ex_list

def getCondition(arr, exclude):
    cond_bool = np.ones(arr.shape, dtype=bool)
    for ex in exclude:
        cond_bool[ex[0]:ex[1]] = False
      
    return cond_bool
    
def getVariance(arr,condition):
    var1 = np.var(arr[np.logical_not(condition)])
    var2 = np.var(arr[condition])
    mn1 = np.mean(arr[np.logical_not(condition)])
    mn2 = np.mean(arr[condition])
    return var1,var2,mn1,mn2
     
def getTriangle(arr, left, right):
    # more then 2 distinct values
    assert(left >= 0 and left < len(arr))
    assert(right >= 0 and right < len(arr))
    assert(left < right)
    assert(len(set(arr)) > 1)

    arr = arr.flatten()    

    lm,rm = getMean(arr,left,right)    
    argmx = np.nanargmax(arr[left:right])+left
    slope = getSlope(arr,lm,rm,argmx)
    diff = arr - slope
    li,ri = getIndexes(diff,left,right,argmx)
    
    return li,ri,argmx

def getMean(arr, left, right):
    if left == 0:
        l_mean = arr[0]
    else:
        l_mean = np.mean(arr[:left])
        
    if right == len(arr)-1:
        r_mean = arr[-1]
    else:
        r_mean = np.mean(arr[right:])
        
    return l_mean,r_mean

def getArgmax(arr,left,right,exclude=None):
    if exclude is None:
        return np.argmax(arr[left:right])+left
    else:
        return np.argmax(arr[exclude][left:right])+left
        
def getSlope(arr,left_mean,right_mean,argmx):
    slope = np.zeros(len(arr))    
    
    # make left and right slope (triangle)
    slope[:argmx] = np.linspace(left_mean,arr[argmx],
                                num=argmx, endpoint=True)     
                                
    slope[argmx:] = np.linspace(arr[argmx],right_mean,
                                num=len(arr)-argmx, endpoint=True)  
    
    return slope

def getIndexes(arr,left,right,argmx):
    li = np.nanargmin(arr[left:argmx])+left
    ri = np.nanargmin(arr[argmx:right])+argmx
    
    return li,ri
    
def triangle(arr, left, right, exclude=None):
    # more then 2 distinct values

    assert(left >= 0 and left < len(arr))
    assert(right >= 0 and right < len(arr))
    assert(left < right)
    
    if exclude is None:
        exclude = []
        
    arr = arr.flatten()    

    slope = np.zeros(len(arr))
    diff = np.zeros(len(arr))

    if len(set(arr)) > 1:
        if left == 0:
            l_mean = arr[0]
        else:
            l_mean = np.median(arr[:left])
            
        if right == len(arr)-1:
            r_mean = arr[-1]
        else:
            r_mean = np.median(arr[right:])
                
        argmx = np.argmax(arr[left:right])
    
        mx = arr[left+argmx]  
        lsl = left+argmx
        rsl = len(arr)- lsl
        
        # make left and right slope (triangle)
        slope[:lsl] = np.linspace(l_mean,mx,num=lsl, endpoint=True)                                   
        slope[lsl:] = np.linspace(mx,r_mean,num=rsl, endpoint=True)   
        
        # substract slopes from sections
        diff = arr - slope
        
        # more then 2 distinct values
        if len(set(diff)) > 1:
            
            if argmx > 0:
                li = np.argmin(diff[left:lsl])+left
            else:
                li = left
                
            if argmx < right-left:
                ri = np.argmin(diff[lsl:right])+lsl
            else:
                ri = right
            
            return li,ri,diff,slope
    
    return -1,-1,diff,slope
    
def verify( arr, li, ri, left, right):

    if li > 0 and ri > 0 and li < ri:
        mm = np.mean(arr[li:ri])
        
        if left < li:
            lm = max(arr[left:li])
        elif left == li:            
            lm = arr[li]      
        else:
            lm = left
            print("LEFT > LI")
            
        if right > ri:
            rm = max(arr[ri:right])            
        elif right == ri:
            rm = arr[ri]
        else:
            rm = right
            print("RIGHT < RI")
            
        mx = max((lm,rm)) 

        if mm > mx:
            return True

    return False
        
def findDeposition( image, left=1, right=-1 ):
    
    R,G,B = cv2.split(image)       
    channels = [R,G,B]        

    sses = [template(ch,left=left,right=right) for ch in channels]
    nsse = [normalize(sse,left=left,right=right) for sse in sses]
    total = euclidean(nsse)
    
    li,ri,d,s = triangle(total,left=left,right=right)
    
    return [li,ri,total,s]

def findDeposition2( image, left=1, right=-1 ):
    
    R,G,B = cv2.split(image)       
    channels = [R,G,B]        

    sses = [template(ch,left=left,right=right) for ch in channels]
    nsse = [normalize(sse,left=left,right=right) for sse in sses]
    total = euclidean(nsse)
    
    while(comb_var <  var):
        var = comb_var
        li,ri,d,s = triangle(total,left=left,right=right)        
        comb_var = combinedVariance(total,li,ri)
        
    
    lvi,rvi,mean,mx = verify(total,li,ri)

    return lvi,rvi,total,s
    
def findLineEdge( channel,left,right, low_t, high_t, side="out", below=0):

    if left == 0 and right == channel.shape[1]-1:    
        ch_side = channel
        side = "total"
    else:
        if side is "out":
            ch_side = np.hstack((channel[:,:left],channel[:,right:]))     
        elif side is "in":
            ch_side = channel[:,left:right]       
    
    copy = np.copy(ch_side)
    
    rng,t = threshold( ch_side ,t_min=low_t, t_max=high_t, below=below)
    topt = rng[np.argmin(t)]
    
    copy[copy < topt] = 0
    row_mean = np.mean(copy,axis=1)    
    li,ri,d,s = triangle(row_mean, below, len(row_mean)-1)       
    mi = np.argmax(row_mean)
    mid = midval(row_mean,li,mi)
    
    if side is "out":
        channel[:,:left] = copy[:,:left]
        channel[:,right:] = copy[:,left:]
    elif side is "in":
        channel[:,left:right] = copy
            
    return [li,mid,mi,row_mean,topt,rng,t]
    
def template(channel, left=1, right=-1):
    assert(len(channel.shape) == 2)   
        
    lc = channel[ :, :left ]
    rc = channel[ :, right: ] 
    tc = np.hstack( ( lc, rc ) )    
    tmp = np.mean( tc, axis=1)   
    s2 = ( ( channel.T - tmp).T )**2
    ss2 = np.sum(s2, axis=0)
    return ss2

def normalize(arr, left=1, right=-1):
    assert(len(arr) > 1)
    
    sides = np.hstack((arr[:left],arr[right:]))
    std = np.std( sides, ddof=1)  
    mn = np.mean( sides )             
    norm = (arr-mn) / std
    return norm
    
def euclidean(arrs):
    total_error = np.zeros(len(arrs[0]))
    for arr in arrs:
        total_error += arr**2
            
    sqrt = total_error**0.5    
    
    return sqrt

def histograms(arr_list):
    mn1 = np.amin(arr_list[0])
    mx1 = np.amax(arr_list[0])
    
    for arr in arr_list:
        mn = np.amin(arr)
        mx = np.amax(arr)
        if mn < mn1:
            mn1 = mn
        if mx > mx1:
            mx1 = mx

    lst = [((arr-mn)*255/mx).astype(np.uint8) for arr in arr_list] 
    
    hist_list = [np.bincount(l) for l in lst]
    rng = np.linspace(mn,mx,num=len(hist_list[0]))
    return rng,hist_list
    
def threshold(channel,t_min,t_max, below):   

    t_ch = channel
        
    t_rng = np.arange(t_min,t_max)
    t_depth = np.array([th(t_ch,t,below) for t in t_rng])

    return t_rng,t_depth
    
def th(channel,threshold, below):
    channel[channel < threshold] = 0
    m_row = np.mean(channel,axis=1)
    li,ri,d,s = triangle(m_row, below, len(m_row)-1)
    return d[li] 

    
def midval( arr, li, ri, side="left"):
    
    mid_val = (arr[ri] - arr[li]) / 2 + arr[li]     
    
    mi = 0
    for i in range(ri-li):
        if arr[li+i] > mid_val:
            mi = li+i
            break   
        
    return mi
    
