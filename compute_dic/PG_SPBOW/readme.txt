========================================================================
Spatial Pyramid Code and Bag of Words Matlab Code
Created by Piji Li (peegeelee@gmail.com)  
Blog: http://www.zhizhihu.com
QQ: 379115886
IRLab. : http://ir.sdu.edu.cn     
Shandong University,Jinan,China
10/1/2010

Some code are from:

S. Lazebnik, C. Schmid, and J. Ponce, "Beyond Bags of Features: Spatial 
Pyramid Matching for Recognizing Natural Scene Categories," CVPR 2006.

========================================================================


Just modify the main.m: rootpath='your PG_SPBOW path, and then run it.

The BOW and Dictionary is in the dir:/data/global, size of BOW_sift.mat is (DictionarySize * #images).
Size of dictionary.mat is (DictionarySize *  dim of features).spatial_pyramid.mat is the Spatial Pyramid BoW.


In /data/local is the sift features for each images. 
