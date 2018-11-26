import numpy

'''朴素贝叶斯分类算法'''
# https://www.cnblogs.com/muchen/p/6305397.html
# 朴素贝叶斯分类算法常常用于文档的分类，而且实践证明效果挺不错的。
# 在说明原理之前，先介绍一个叫‘词向量’的概念。
# 它一般是一个布尔类型的集合，该集合中每个元素都表示其对应的单词是否在文档中出现。
# 比如说，词汇表只有三个单词：'apple', 'orange', 'melo'，某文档中，apple和melo出现过，那么其对应的‘词向量’就是 {1, 0, 1}。
# 这种模型通常称为‘词集模型’，如果词向量元素是整数类型，每个元素表示相应单词在文档中出现的次数(0表示不出现)，那这种模型就叫做‘词袋模型’。

# 如下部分代码可用于由文档构建词向量以及测试结果：

'''创建测试数据'''
def loadDataSet():
    # 这组数据是从斑点狗论坛获取的
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1表示带敏感词汇
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

'''创建词汇表'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:# 遍历文档列表
        # 首先将当前文档的单词唯一化，然后以交集的方式加入到保存词汇的集合中。
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''将文档转换为词向量'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)# 构建vocabList长度的0向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print(str(word) + "不在词汇表当中")
    return returnVec

'''测试'''
def test(): 
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(str(myVocabList))
    print(str(listOPosts[0]))
    print(str(setOfWords2Vec(myVocabList, listOPosts[0])))
# myVocabList: ['to', 'garbage', 'quit', 'not', 'him', 'ate', 'mr', 'worthless', 'so', 'love', 'is', 'steak', 'dog', 'I', 'please', 'help', 'flea', 'stop',
# 'maybe', 'take', 'park', 'dalmation', 'how', 'my', 'posting', 'food', 'licks', 'buying', 'cute', 'has', 'problems', 'stupid']
# listOPosts[0]: ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'
# returnVec: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]


'''算法原理'''
# 不论是用于训练还是分类的文档，首先一致处理为词向量。
# 通过贝叶斯算法对数据集进行训练，从而统计出所有词向量各种分类的概率。
# 对于待分类的文档，在转换为词向量之后，从训练集中取得该词向量为各种分类的概率，概率最大的分类就是所求分类结果。


'''训练算法剖析：如何计算某个词向量的概率'''
# 由贝叶斯公式可知，某词向量X为分类 Ci 的概率可用如下公式来进行计算：P(Ci | W) = [P(W | Ci) * P(Ci)] / P(W)
# p(ci)表示该文档为分类ci的概率；
# p(w)为该文档对应词向量为w的概率；这两个量是很好求的，这里不多解释。
# 关键要解决的是 p(w|ci)，也即在文档为分类 ci 的条件下，词向量为w的概率。

# 这里就要谈到为什么本文讲解的算法名为 "朴素" 贝叶斯。所谓朴素，就是整个形式化过程只做最原始假设。
# 也就是说，假设不同的特征是相互独立的。但这和现实世界不一致，也导致了其他各种形形色色的贝叶斯算法。
# 在这样的假设前提下： p(w|ci) = p(w0|ci) * p(w1|ci) * p(w2|ci) * .... * p(wn|ci)。
# 而前面提到了w是指词向量，这里wn的含义就是词向量中的某个单词。
# 可使用如下伪代码计算条件概率 p(wn|ci)：
'''
对每篇训练文档:
     对每个类别：
         增加该单词计数值
         增加所有单词计数值
     对每个类别:
         对每个单词:
             将该单词的数目除以单词总数得到条件概率
返回所有单词在各个类别下的条件概率
'''


'''朴素贝叶斯分类算法'''
#=============================================
#    输入:
#        trainMatrix:     文档矩阵
#        trainCategory:       分类标签集
#    输出:
#        p0Vect:    各单词在分类0的条件下出现的概率
#        p1Vect:    各单词在分类1的条件下出现的概率
#        pAbusive:    文档属于分类1的概率
#============================================= 
def trainNB0(trainMatrix,trainCategory):  
    # 文档个数
    numTrainDocs = len(trainMatrix)
    # 文档词数
    numWords = len(trainMatrix[0])
    # 文档属于分类1的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 属于分类0的词向量求和
    p0Num = numpy.ones(numWords);
    # 属于分类1的词向量求和
    p1Num = numpy.ones(numWords)
    
    # 分类 0/1 的所有文档内的所有单词数统计
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):   # 遍历各文档
        
        # 若文档属于分类1
        if trainCategory[i] == 1:
            # 词向量累加
            p1Num += trainMatrix[i]
            # 分类1文档单词数累加
            p1Denom += sum(trainMatrix[i])
            
        # 若文档属于分类0
        else:
            # 词向量累加
            p0Num += trainMatrix[i]
            # 分类0文档单词数累加
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = numpy.log(p1Num/p1Denom)
    p0Vect = numpy.log(p0Num/p0Denom)
    
    return p0Vect, p1Vect, pAbusive


#=============================================
#    输入:
#        vec2Classify:     目标对象的词向量的数组形式
#        p0Vect:    各单词在分类0的条件下出现的概率
#        p1Vect:    各单词在分类1的条件下出现的概率
#        pClass1:  文档属于分类1的概率
#    输出:
#        分类结果 0/1
#=============================================
'''完成贝叶斯公式剩余部分得到最终分类概率'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 为分类1的概率
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)
    # 为分类0的概率
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
'''测试'''
def test2():    
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    
    # 创建文档矩阵
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
    # 对文档矩阵进行朴素贝叶斯分类并返回各单词在各分类条件下的概率及文档为类别1的概率
    p0V,p1V,pAb = trainNB0(numpy.array(trainMat),numpy.array(listClasses))
    
    # 测试一
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(str(testEntry) + '分类结果: ' + str(classifyNB(thisDoc,p0V,p1V,pAb)))
    
    # 测试二
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(str(testEntry) + '分类结果: ' + str(classifyNB(thisDoc,p0V,p1V,pAb)))
'''
['love', 'my', 'dalmation']分类结果: 0
['stupid', 'garbage']分类结果: 1
'''




















    
    
