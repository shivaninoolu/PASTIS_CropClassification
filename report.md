### PASTIS Crop Classification - 

### Baseline U-Net with Sentinel 10 bands \& U-TAE Sentinel 10 bands + Vegetation Indices

==========================================================================================================================================================================

#### **Dataset**

**--------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

**PASTIS** (Panoptic Agricultural Satellite TIme Series): Sentinel-2 optical (10 bands) time series (46 dates) with pixel-wise crop-type labels (18 + 1 void = 19).



**Data\_S2**:		(46, 10, 128, 128)		46 acquisition dates, 10 spectral bands, 128×128 pixel patch

**ANNOTATION**:		(1, 128, 128)			per-pixel class label, 0–18 = crop/land classes, 19 = Void

**metadata.geojson**:	102 samples divided into 5 folds, parcel, acquisition dates, tile id



In the temporal sequence all 46 dates should be available. So to handle the bad/cloud dates without breaking the tensor shape cloud I have followed interpolation instead of deleting the dates. A pre-assigned 1–5 grouping that is provided is used for 5-fold cross-validation where one fold is held out as validation and the other four are pooled for training, geographically, so a training patch and a validation patch from the same field/tile are not likely to be right next to each other. This matters because in remote-sensing data adjacent patches are spatially correlated (same soil, weather, crop rotation), so a simple random split would leak information.



================================================================================

PASTIS DATASET INSPECTION

================================================================================



FILE COUNTS

\--------------------------------------------------------------------------------

Number of S2 files     : 102

Number of target files : 102



PAIRING CHECK

\--------------------------------------------------------------------------------

S2 sample IDs      : 102

Target sample IDs  : 102

Paired samples     : 102



================================================================================

INPUT SHAPE SUMMARY

================================================================================

(46, 10, 128, 128): 102 samples



================================================================================

TARGET SHAPE SUMMARY

================================================================================

(1, 128, 128): 102 samples





#### **Preprocessing (Clouse handling, Normalisation)**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**



* Brightness and NDVI are used together to identify possible clouds. Clouds are usually very bright and have very little vegetation signal. Bare soil can also be bright with low NDVI, so neither measure is used alone.
* Next we check whether the brightness is unusual in two ways: compared with other pixels in the same image (spatial check) and compared with the same pixel on other dates (temporal check). A 97th-percentile threshold is used to detect only very strong brightness spikes, reducing the chance of removing real crop information.
* If a date has too much cloud contamination, the whole date is treated as unreliable instead of keeping noisy data. Rather than simply removing that date, the missing values are linearly interpolated from the nearest valid dates. This keeps the time series continuous and helps preserve the natural crop growth/phenological curve.
* Normalization: Per-channel z-score standardization ((x - mean) / std) Sentinel-2 bands and the derived indices are of different numeric scales, so unnormalized inputs would make some channels dominate.



SENTINEL-2 BAND STATISTICS

================================================================================

&#x20;Idx | Band  |          Min |          Max |         Mean |          Std

\--------------------------------------------------------------------------------

&#x20;  0 | B2    |        0.000 |    20716.000 |     1243.564 |     1910.807

&#x20;  1 | B3    |        0.000 |    18985.000 |     1451.851 |     1839.640

&#x20;  2 | B4    |        0.000 |    16610.000 |     1498.940 |     1922.610

&#x20;  3 | B5    |        0.000 |    16161.000 |     1835.883 |     1800.475

&#x20;  4 | B6    |        0.000 |    15830.000 |     2847.227 |     1641.834

&#x20;  5 | B7    |        0.000 |    15450.000 |     3216.133 |     1653.507

&#x20;  6 | B8    |        0.000 |    15584.000 |     3311.183 |     1663.657

&#x20;  7 | B8A   |        0.000 |    14973.000 |     3481.729 |     1641.112

&#x20;  8 | B11   |     -601.000 |    12270.000 |     2557.232 |     1343.902

&#x20;  9 | B12   |        0.000 |    10940.000 |     1717.557 |     1192.654

================================================================================





#### **Feature Engineering (using Vegetation Indices)**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

Reflectance bands alone are not the most efficient way to represent crop phenology. So 4 vegetation indices are chosen:

* **NDVI** - general vegetation vigor/greenness. saturates in dense canopies.
* **EVI** - same as NDVI but includes a blue-band correction term that reduces atmospheric and soil-background noise and sensitive
at high biomass
* **LSWI** - uses SWIR band — sensitive to canopy/soil water content. Can separate crops with similar NDVI patterns.
* **NDRE** - Uses the red-edge band — sensitive to chlorophyll content. 

The 4 indices jointly cover vigor, biomass at saturation, water status, and chlorophyll.



#### **Feature Selection**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**



We start with 10 Sentinel-2 bands + 4 vegetation indices = 14 channels. Since some channels contain similar information, using all 14 can add unnecessary complexity.



**Fisher score** is used to identify channels that best distinguish between different crop classes.

**Correlation pruning** removes channels that provide almost the same information as already-selected channels.

Selection is done only using training-fold pixels, so there is no information leakage from validation data.

The final selection is limited to 8 channels to keep the model compact and reduce overfitting, while still retaining useful spectral information.

The baseline uses all 10 raw bands without selection, so it provides a reference for measuring the benefit of feature engineering and selection.



**FEATURE SELECTION**

\------------------------------------------------------------

&#x20; LSWI  Fisher=4207674.759360  SELECTED

&#x20; NDVI  Fisher=1.177115  SELECTED

&#x20; NDRE  Fisher=0.809618  

&#x20;   B4  Fisher=0.551861  SELECTED

&#x20;  B12  Fisher=0.457884  SELECTED

&#x20;   B5  Fisher=0.413179  SELECTED

&#x20;   B6  Fisher=0.386190  SELECTED

&#x20;  B8A  Fisher=0.366492  

&#x20;   B7  Fisher=0.361654  

&#x20;   B8  Fisher=0.354478  SELECTED

&#x20;  B11  Fisher=0.287996  SELECTED

&#x20;   B3  Fisher=0.258294  

&#x20;   B2  Fisher=0.138854  

&#x20;  EVI  Fisher=0.002428  



Selected features:

\['LSWI', 'NDVI', 'B4', 'B12', 'B5', 'B6', 'B8', 'B11']

Input channels: 8





#### **Class Weights/ Class Imbalance handling**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**



================================================================================

CLASS STATISTICS

================================================================================

&#x20; ID | Class Name                     |          Pixels |   % Pixels |  Samples

\--------------------------------------------------------------------------------

&#x20;  0 | Background                     |         506,141 |   30.2867% |      102

&#x20;  1 | Meadow                         |         296,576 |   17.7466% |      101

&#x20;  2 | Soft winter wheat              |         199,108 |   11.9143% |       95

&#x20;  3 | Corn                           |         196,704 |   11.7705% |       98

&#x20;  4 | Winter barley                  |          57,381 |    3.4336% |       64

&#x20;  5 | Winter rapeseed                |          92,912 |    5.5597% |       65

&#x20;  6 | Spring barley                  |           3,472 |    0.2078% |        9

&#x20;  7 | Sunflower                      |           9,579 |    0.5732% |       18

&#x20;  8 | Grapevine                      |           1,345 |    0.0805% |        1

&#x20;  9 | Beet                           |               0 |    0.0000% |        0

&#x20; 10 | Winter triticale               |          15,215 |    0.9104% |       25

&#x20; 11 | Winter durum wheat             |             725 |    0.0434% |        3

&#x20; 12 | Fruits, vegetables, flowers    |           3,229 |    0.1932% |        6

&#x20; 13 | Potatoes                       |             158 |    0.0095% |        1

&#x20; 14 | Leguminous fodder              |          12,634 |    0.7560% |       27

&#x20; 15 | Soybeans                       |         121,560 |    7.2740% |       80

&#x20; 16 | Orchard                        |              75 |    0.0045% |        1

&#x20; 17 | Mixed cereal                   |           5,220 |    0.3124% |       13

&#x20; 18 | Sorghum                        |           5,047 |    0.3020% |        7

&#x20; 19 | Void label                     |         144,087 |    8.6219% |      102



================================================================================





In our dataset background and common crops are covering most pixels, while some crops have very few pixels. Without weighting, the model may focus mainly on common classes and ignore rare ones.

**Square-root inverse frequency** is used to give more importance to rare classes without making their weights too extreme.

Weights are calculated separately for each fold using only training pixels, avoiding validation-data leakage.

Weighting is applied only to the training loss. Validation uses the normal unweighted loss so it reflects the natural dataset distribution.

mIoU is the main balanced metric, ensuring that performance on minority classes is also considered.





#### **Model**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

**Baseline model** —

* raw Sentinel-2 bands,
* a lightweight Conv3D + U-Net decoder.

**U-TAE + Indices** —

* Sentinel-2 bands + 4 vegetation indices,
* Fisher-score feature selection,
* U-TAE model.
* 

Both scenarios use the same lightweight U-Net decoder chosen because it is simple, effective for pixel-level segmentation and suitable for the small dataset.

Baseline – TemporalSpatialCNNUNet: Uses 3D convolutions to process the 46-date sequence and then averages the dates. It was chosen as a simple reference model without learned temporal weighting.

UTAE – It uses a shared spatial encoder for each date and temporal attention to learn which dates are more useful for distinguishing crop types. It was chosen to better capture the most informative periods in the crop time series.



This comparison helps determine whether learned temporal attention provides an advantage over simple temporal averaging.





#### **TRAINING**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

→ Per (scenario, fold) 

→ split via 

→ feature selection on the training fold (UTAE model only)

→ compute per-channel mean/std from the training fold

→ channels 10 for baseline, up to 8 for utae\_indices

→ load that fold's class weights 

→ train with Adam + a weighted CrossEntropyLoss.

* **ReduceLROnPlateau on validation mIoU** (not loss) — mIoU is the metric that actually reflects segmentation quality across classes, so reducing the learning rate when that stalls is more aligned with the end goal than watching the loss, which is dominated by the classes at that instance hiving the most weighted error.
* **Early stopping on validation mIoU with patience 10** — given a dataset this size, letting training run indefinitely risks overfitting to the training fold well past the point where validation performance is actually improving, stopping on a plateau of the target metric is more useful here than a fixed epochs.
* **Checkpoints** — save everything needed to reproduce evaluation identically later (mean/std, selected features, class weights, scenario)





#### **Cross Validation**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

→ Looping through folds 1–5 for a given scenario

→ training then evaluating each fold and aggregating per-fold and per-class metrics into mean ± std across folds. 

→ 5-fold spatial CV.



================================================================================

5-FOLD CROSS-VALIDATION COMPLETE: baseline

================================================================================

Fold 1: Overall Accuracy=0.7725, mIoU=0.3767, best epoch=20

Fold 2: Overall Accuracy=0.8175, mIoU=0.3392, best epoch=26

Fold 3: Overall Accuracy=0.8417, mIoU=0.4661, best epoch=46

Fold 4: Overall Accuracy=0.8072, mIoU=0.4120, best epoch=24

Fold 5: Overall Accuracy=0.8244, mIoU=0.4472, best epoch=38



Mean Overall Accuracy = 0.8127 ± 0.0230

Mean mIoU             = 0.4082 ± 0.0461



&#x09;					MEAN		STD

\-------------------------------------------------------------------

class\_1\_Meadow				0.784284625	0.056671387

class\_2\_Soft\_winter\_wheat		0.901153061	0.033445062

class\_3\_Corn				0.875669465	0.034950611

class\_4\_Winter\_barley			0.700144539	0.207474905

class\_5\_Winter\_rapeseed			0.954066153	0.021194939

class\_6\_Spring\_barley			0.011428571	0.022857143

class\_7\_Sunflower			0.706268366	0.137646216

class\_8\_Grapevine			0	0

class\_9\_Beet				nan	nan

class\_10\_Winter\_triticale		0.538998031	0.279865715

class\_11\_Winter\_durum\_wheat		0	0

class\_12\_Fruits\_vegetables\_flowers	0	0

class\_13\_Potatoes			0	0

class\_14\_Leguminous\_fodder		0.359209325	0.090997761

class\_15\_Soybeans			0.859149135	0.064518648

class\_16\_Orchard			0	0

class\_17\_Mixed\_cereal			0.059486628	0.105618819

class\_18\_Sorghum			0.03307393	0.057285727



performance drops substantially for rare classes. Spring barley, Mixed cereal, and Sorghum have very low recall, while several extremely rare classes have 0% detection.



This is consistent with the dataset distribution: the four largest classes—Background, Meadow, Soft winter wheat, and Corn account for about 71.7% of the labelled pixels, while several classes have fewer than 1,000 pixels or even only a single sample. Beet has no pixels in the dataset.

The 40.82% mIoU and per-class results reveal the much larger variation between common and rare classes.



================================================================================

5-FOLD CROSS-VALIDATION COMPLETE: utae\_indices

================================================================================

Fold 1: Overall Accuracy=0.8111, mIoU=0.3976, best epoch=46

Fold 2: Overall Accuracy=0.7864, mIoU=0.3043, best epoch=23

Fold 3: Overall Accuracy=0.8342, mIoU=0.4059, best epoch=29

Fold 4: Overall Accuracy=0.8325, mIoU=0.4548, best epoch=37

Fold 5: Overall Accuracy=0.8355, mIoU=0.4297, best epoch=38



Mean Overall Accuracy = 0.8199 ± 0.0190

Mean mIoU             = 0.3985 ± 0.0511



&#x09;					MEAN		STD

\-------------------------------------------------------------------

class\_0\_Background			0.808210343	0.028727966

class\_1\_Meadow				0.751868544	0.040336899

class\_2\_Soft\_winter\_wheat		0.930854339	0.020750445

class\_3\_Corn				0.909596817	0.025875927

class\_4\_Winter\_barley			0.793196068	0.030407715

class\_5\_Winter\_rapeseed			0.959413374	0.014821162

class\_6\_Spring\_barley			0.011384615	0.022769231

class\_7\_Sunflower			0.551888142	0.324180999

class\_8\_Grapevine				0		0

class\_9\_Beet					nan		nan

class\_10\_Winter\_triticale		0.461836342	0.164618087

class\_11\_Winter\_durum\_wheat			0		0

class\_12\_Fruits\_vegetables\_flowers	0.040697674	0.07049044

class\_13\_Potatoes				0		0

class\_14\_Leguminous\_fodder		0.485541326	0.065092987

class\_15\_Soybeans			0.830226649	0.059159453

class\_16\_Orchard				0		0

class\_17\_Mixed\_cereal			0.035608309	0.071216617

class\_18\_Sorghum				0		0



Performance is strong and relatively stable for well-represented classes such as winter rapeseed, soft winter wheat, corn, soybeans and winter barley. In contrast, recognition of rare classes was poor, with several classes achieving zero recall. Less frequent classes such as sunflower and winter triticale also exhibited substantial variation between folds. These results indicate that the selected spectral features combined with temporal attention are effective for several dominant crop classes, but the model remains strongly affected by the severe class imbalance and limited sample representation of minority crop classes.



#### **Analysis \& Interpretation**

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------------**



* UTAE was selected because the PASTIS dataset contains multi-temporal Sentinel-2 observations. Its temporal attention can learn which dates are more informative for crop discrimination
* Selected spectral bands and indices were intended to reduce redundancy and retain informative features
* UTAE + indices achieved 0.8199 accuracy compared with 0.8127 for the baseline, but mIoU decreased from 0.4082 to 0.3985.
* The higher accuracy suggests better prediction of many pixels, particularly common classes, while the lower mIoU indicates that performance is not consistently balanced across all classes
* Winter rapeseed (0.959), soft winter wheat (0.931), corn (0.910), soybeans (0.830), and winter barley (0.793) achieved relatively high recall
* Spring barley, Grapevine, Winter durum wheat, Potatoes, Orchard, Mixed cereal, and Sorghum had very low or zero recall
* **Class confusion**: Similar crops, particularly different cereal types, can have similar spectral signatures during certain growth stages, making them difficult to distinguish
* **Class imbalance:** Common classes dominate the dataset, while several minority classes have very few pixels and samples.
* The heterogeneous agricultural landscape contains different crop types, field sizes, and uneven class distributions, which can also contribute to variation between cross-validation folds.
* The primary limitation of the UTAE-index model is its poor recognition of severely underrepresented crop classes. 
* Improvement needs to be done on minority-class representation through class sampling, augmentation and additional labelled data where available. 
* Since UTAE exploits temporal information, analysis of temporal attention and identification of the most discriminative phenological periods could further improve crop discrimination. 
* Alternative loss functions, feature configurations and class-balancing strategies can be evaluated while maintaining the same cross-validation. 





















