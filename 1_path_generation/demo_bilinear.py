#arg1 is term1
#arg2 is term2
#arg3 is the way to get the score, valid input including
#all, max, topfive, sum
#[SymbolOf, CreatedBy, MadeOf, PartOf, HasLastSubevent, HasFirstSubevent, Desires, CausesDesire,
#DefinedAs, HasA, ReceivesAction, MotivatedByGoal, Causes, HasProperty, HasPrerequisite, 
#HasSubevent, AtLocation, IsA, CapableOf, UsedFor]
#case insensitive for the third argument
import pickle
import numpy as np
import sys
import math

def getVec(We,words,t):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]],:]
    else:
        vec = We[words['UUUNKKK'],:]
        print('can not find corresponding vector:',array[0].lower())
    for i in range(len(array)-1):
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]],:]
        else:
            print('can not find corresponding vector:',array[i+1].lower())
            vec = vec + We[words['UUUNKKK'],:]
    vec = vec/len(array)
    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def score(term1,term2,words,We,rel,Rel,Weight,Offset,evaType):
    v1 = getVec(We,words,term1)
    v2 = getVec(We,words,term2)
    result = {}
    """

    del_rels = ['HasPainCharacter', 'HasPainIntensity', 'LocationOfAction', 'LocatedNear',
    'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
    'NotHasA','NotIsA','NotHasProperty','NotCapableOf']
    
    for del_rel in del_rels:
        print(rel, del_rel.lower())
        del rel[del_rel.lower()]
    """
    for k,v in rel.items():
        v_r = Rel[rel[k],:]
        gv1 = np.tanh(np.dot(v1,Weight)+Offset)
        gv2= np.tanh(np.dot(v2,Weight)+Offset)
    
        temp1 = np.dot(gv1, v_r)
        score = np.inner(temp1,gv2)
        result[k] = (sigmoid(score))

    if(evaType.lower()=='max'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        return result[:1]
    if(evaType.lower()=='topfive'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        return result[:5]
    if(evaType.lower()=='sum'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        total = 0
        for i in result:
            total = total + i[1]
        return total
    if(evaType.lower()=='all'):
        result = sorted(result.items(), key=lambda x: x[1], reverse = True)
        return result
    else:
        tar_rel = evaType.lower()
        if result.get(tar_rel) == None:
            print('illegal relation, please re-enter a valid relation')
            return 'None'
        else:
            print(tar_rel,'relation score:',result.get(tar_rel))
            return result.get(tar_rel)

class Scorer():
    def __init__(self):
        model = pickle.load(open("/home/hyx/laq/ckbc-demo/Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pikle", "rb"), encoding='iso-8859-1')

        self.Rel = model['rel']
        self.We = model['embeddings']
        self.Weight = model['weight']
        self.Offset = model['bias']
        self.words = model['words_name']
        self.rel = model['rel_name']
        del_rels = ['HasPainCharacter', 'HasPainIntensity', 'LocationOfAction',
    'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
    'NotHasA','NotIsA','NotHasProperty','NotCapableOf']

        for del_rel in del_rels:
            del self.rel[del_rel.lower()]

    def gen_score(self, e1, e2, method='all'):
        e1 = "_".join(e1.split())
        e2 = "_".join(e2.split())
        result=score(e1, e2,self.words,self.We,self.rel,self.Rel,self.Weight,self.Offset,method)
        return result
