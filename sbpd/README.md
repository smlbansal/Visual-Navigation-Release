We follow the method outlined in ["Cognitive Mapping and Planning for Visual Navigation"](https://github.com/tensorflow/models/tree/master/research/cognitive_mapping_and_planning) to download the SD3DIS data. Please follow the steps below

### Download and preprocess the meshes

1.  Download the data from the [dataset website](http://buildingparser.stanford.edu/dataset.html).
    1.  [Raw meshes](https://goo.gl/forms/2YSPaO2UKmn5Td5m2). We need the meshes
        which are in the noXYZ folder. Download the tar files and place them in
        the `stanford_building_parser_dataset_raw` folder. Download `area_1_noXYZ.tar`, `area_3_noXYZ.tar`, `area_4_noXYZ.tar`, `area_5a_noXYZ.tar`, `area_6_noXYZ.tar`.
        

2.  Preprocess the data.

    1.  Extract meshes using `bash preprocess_meshes.sh`. After
        this `ls stanford_building_parser_dataset/mesh` should have 6
        folders `area1`, `area3`, `area4`, `area5a`, `area5b`, `area6`, with
        textures and obj files within each directory. Note: This script assumes you have downloaded data for all 6 areas, if you downloaded data for a subset of the areas, modify preprocess_meshes.sh.
        
       

### Download Precomputed Traversable Maps (~160 KB).
For experiment reproducibility we include precomputed traversable maps of each of the 6 SBPD areas. Inside the folder stanford_building_parser_dataset, do the following:
```
# To download the data via the command line run the following
pip install gdown
gdown https://drive.google.com/uc?id=1rkCVSE3JiJShykIIq2ss3dIL3cUrIeKG

# To download the data via your browser visit the following url
https://drive.google.com/a/berkeley.edu/file/d/1rkCVSE3JiJShykIIq2ss3dIL3cUrIeKG/view?usp=sharing

# Unzip the file LB_WayPtNav_Data.tar.gz
tar -zxf traversibles.tar.gz
```

If installed correctly you should have a directory stucture as follows:
```
stanford_building_parser_dataset/
    - mesh/
        - area1/
        Optionally
        - area2/
        ...
        - area6
    - traversibles/
        - area1/
        ...
        - area6/
```
