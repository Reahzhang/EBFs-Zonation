//-----------------------------------------------获取影像行列号-------------------------------------------------------
var collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")  
            .filterDate('2020-1-02', '2020-01-30')
            .filterBounds(table);
Map.addLayer(table)
// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

var images_list = ee.List(collection.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list'));
var p_list = ee.List([]);
var r_list = ee.List([]);

var path_select = images_list.map(function(num){
  var image = ee.Image(collection.filter(ee.Filter.eq("system:index", num)).first());
  var properties = ee.Dictionary(ee.Dictionary(ee.Algorithms.Describe(image)).get('properties'));
  var wrs_path = properties.get('WRS_PATH');
  p_list = p_list.add(wrs_path);
  return p_list;
});

var row_select = images_list.map(function(num){
  var image = ee.Image(collection.filter(ee.Filter.eq("system:index", num)).first());
  var properties = ee.Dictionary(ee.Dictionary(ee.Algorithms.Describe(image)).get('properties'));
  var wrs_row = properties.get('WRS_ROW');
  r_list = r_list.add(wrs_row)
  return r_list;
});

var path_list = ee.List(path_select);
var row_list = ee.List(row_select);
//print('path_list_length', path_list.length());
//print('row_list_length', row_list.length());

path_list = path_list.distinct();
row_list = row_list.distinct();
//print('path_list_distinct_length', path_list.length());
//print('row_list_distinct_length', row_list.length());

//-------------------------------------------------影像选择-----------------------------------------------------------
var image_select = function(p) {
  var filtered_col = row_list.map(function(r) {
    var filtered = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                    .filterDate('2019-11-01', '2020-03-31')            
                    .filter(ee.Filter.eq('WRS_PATH', ee.List(p).get(0)))
                    .filter(ee.Filter.eq('WRS_ROW', ee.List(r).get(0)))
                    .sort('CLOUD_COVER')
                    .first();
    return filtered;
  })
  return filtered_col;
}

var selected_images = ee.ImageCollection(path_list.map(image_select).flatten()).filterBounds(table);
//print('selected_images', selected_images);
Map.addLayer(selected_images, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 6000, max: 23000}, 'selected_images', false);

//------------------------------------------------去云-----------------------------------------------------------------
function cloudfree_landsat (image){
  var qa = image.select('QA_PIXEL');
  var cloudsBitMask = 1 << 3;
  var cloudShadowBitMask = 1 << 4;
  var mask = qa.bitwiseAnd(cloudsBitMask).eq(0)
              .and(qa.bitwiseAnd(cloudShadowBitMask).eq(0));
  return image.updateMask(mask)  
}

//当年影像去云
var selected_image = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterDate('2019-11-01', '2020-03-31')
              .filterBounds(table)
              .map(cloudfree_landsat)
              .median();

//三年影像填空值
var year_list = ee.List.sequence(2018,2020);       //插值时间，也可以选
year_list = year_list.map(function(num){
  var n = ee.Number(num).add(1);
  var start_time = ee.Date.fromYMD(num, 11, 1);                   
  var end_time = ee.Date.fromYMD(n, 3, 31);
  var images = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                        .filterDate(start_time, end_time)
                        .filterBounds(table);
  var image = images.map(cloudfree_landsat).median();
  return image;                
});

var cloud_free_image_fill = ee.ImageCollection.fromImages(year_list);
cloud_free_image_fill = cloud_free_image_fill.median();
Map.addLayer(cloud_free_image_fill, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 6000, max: 23000}, 'cloud_free_image_fill', false);

var cloud_free_image = cloud_free_image_fill.blend(selected_image);
Map.addLayer(cloud_free_image, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 6000, max: 23000}, 'cloud_free_image', false);

var cloud_free_image_mosaic = selected_images.map(function(img){
  var mask = img.mask();
  img = img.updateMask(mask.not());
  var image = img.unmask(cloud_free_image)
  return image;
});

var cloud_free_images = ee.ImageCollection(cloud_free_image_mosaic);
cloud_free_images = cloud_free_images.map(applyScaleFactors);
//print('cloud_free_images', cloud_free_images);
Map.addLayer(cloud_free_images, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0.0, max: 0.3}, 'Cloud_free_images', false);

//---------------------------------------------------地形校正----------------------------------------
var scale = 300;

// get terrain layers
var dem = ee.Image('NASA/NASADEM_HGT/001')
var degree2radian = 0.01745;

var terrainCorrection = function(collection) {

  collection = collection.map(illuminationCondition);
  collection = collection.map(illuminationCorrection);

  return(collection);

  // Function to calculate illumination condition (IC). Function by Patrick Burns and Matt Macander 
  function illuminationCondition(img){

  // Extract image metadata about solar position
  var SZ_rad = ee.Image.constant(ee.Number(90).subtract(img.get('SUN_ELEVATION'))).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000)); 
  var SA_rad = ee.Image.constant(ee.Number(img.get('SUN_AZIMUTH')).multiply(3.14159265359).divide(180)).clip(img.geometry().buffer(10000)); 
  // Creat terrain layers
  var slp = ee.Terrain.slope(dem).clip(img.geometry().buffer(10000));
  var slp_rad = ee.Terrain.slope(dem).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000));
  var asp_rad = ee.Terrain.aspect(dem).multiply(3.14159265359).divide(180).clip(img.geometry().buffer(10000));
  
  // Calculate the Illumination Condition (IC)
  // slope part of the illumination condition
  var cosZ = SZ_rad.cos();
  var cosS = slp_rad.cos();
  var slope_illumination = cosS.expression("cosZ * cosS", 
                                          {'cosZ': cosZ,
                                           'cosS': cosS.select('slope')});
  // aspect part of the illumination condition
  var sinZ = SZ_rad.sin(); 
  var sinS = slp_rad.sin();
  var cosAziDiff = (SA_rad.subtract(asp_rad)).cos();
  var aspect_illumination = sinZ.expression("sinZ * sinS * cosAziDiff", 
                                           {'sinZ': sinZ,
                                            'sinS': sinS,
                                            'cosAziDiff': cosAziDiff});
  // full illumination condition (IC)
  var ic = slope_illumination.add(aspect_illumination);

  // Add IC to original image
  var img_plus_ic = ee.Image(img.addBands(ic.rename('IC')).addBands(cosZ.rename('cosZ')).addBands(cosS.rename('cosS')).addBands(slp.rename('slope')));
  return img_plus_ic;
  }

  // Function to apply the Sun-Canopy-Sensor + C (SCSc) correction method to each 
  // image. Function by Patrick Burns and Matt Macander 
  function illuminationCorrection(img){
    var props = img.toDictionary();
    var st = img.get('system:time_start');
    
    var img_plus_ic = img;
    var mask1 = img_plus_ic.select('nir').gt(-0.1);
    var mask2 = img_plus_ic.select('slope').gte(5)
                            .and(img_plus_ic.select('IC').gte(0))
                            .and(img_plus_ic.select('nir').gt(-0.1));
    var img_plus_ic_mask2 = ee.Image(img_plus_ic.updateMask(mask2));
    
    // Specify Bands to topographically correct  
    var bandList = ['blue','green','red','nir','swir1','swir2']; 
    var compositeBands = img.bandNames();
    var nonCorrectBands = img.select(compositeBands.removeAll(bandList));
    
    var geom = ee.Geometry(img.get('system:footprint')).bounds().buffer(10000);
    
    function apply_SCSccorr(band){
      var method = 'SCSc';
      var out = img_plus_ic_mask2.select('IC', band).reduceRegion({
      reducer: ee.Reducer.linearFit(), // Compute coefficients: a(slope), b(offset), c(b/a)
      geometry: ee.Geometry(img.geometry().buffer(-5000)), // trim off the outer edges of the image for linear relationship 
      scale: 300,
      maxPixels: 1000000000
      });  

   if (out === null || out === undefined ){
       return img_plus_ic_mask2.select(band);
       }
  
  else{
      var out_a = ee.Number(out.get('scale'));
      var out_b = ee.Number(out.get('offset'));
      var out_c = out_b.divide(out_a);
      // Apply the SCSc correction
      var SCSc_output = img_plus_ic_mask2.expression(
        "((image * (cosB * cosZ + cvalue)) / (ic + cvalue))", {
        'image': img_plus_ic_mask2.select(band),
        'ic': img_plus_ic_mask2.select('IC'),
        'cosB': img_plus_ic_mask2.select('cosS'),
        'cosZ': img_plus_ic_mask2.select('cosZ'),
        'cvalue': out_c
      });
      
      return SCSc_output;
    }
    
    }
    
    var img_SCSccorr = ee.Image(bandList.map(apply_SCSccorr)).addBands(img_plus_ic.select('IC'));
    var bandList_IC = ee.List([bandList, 'IC']).flatten();
    img_SCSccorr = img_SCSccorr.unmask(img_plus_ic.select(bandList_IC)).select(bandList);
    
    return img_SCSccorr.addBands(nonCorrectBands)
      .setMulti(props)
      .set('system:time_start',st);
  }
}  

var inBands = ee.List(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
var outBands = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']); 

var images_collection = cloud_free_images.select(inBands,outBands);
var images_correction = terrainCorrection(images_collection);
images_correction = images_correction.mosaic();

Map.addLayer(table, {}, 'research_area', false);
Map.addLayer(images_collection, { bands: 'red,green,blue',min: 0.0, max: 0.3}, 'original', false);
Map.addLayer(images_correction, { bands: 'red,green,blue',min: 0.0, max: 0.3}, 'corrected', false);
var image=images_correction
/*var textureimage=images_correction.select("red","nir","blue","green","swir1","swir2");
var input =textureimage.toUint16()
var image = input.glcmTexture();*/

//----------------------------NDVI--------------------------------------------------------------------
var dataset = images_correction.clip(table1);
function NDVI(img){
  var ndvi = img.expression(
    '(NIR-RED)/(NIR+RED)',
    {
      'NIR': img.select('nir'),
      'RED': img.select('red')
    }
  );
  return ndvi;
}

var ndvi_Vis = {
  min: 0.0,
  max: 1.0,
  palette: [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301'
  ],
};

var ndvi = NDVI(image).rename('ndvi').clip(table1);
Map.addLayer(ndvi,  ndvi_Vis, 'ndvi', false);



//---------------------------------------------EVI----------------------------------------------
function EVI(img){
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
    {
      'NIR' : img.select('nir'),
      'RED' : img.select('red'),
      'BLUE': img.select('blue')
    }
  );
  return evi;
}

var evi_Vis = {
  min: 0.0,
  max: 1.0,
  palette: [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301'
  ],
};

var evi = EVI(image).rename('evi').clip(table1);
Map.addLayer(evi, evi_Vis, 'evi', false);
//----------------------------------------------------DEM---------------------------------------------------------
var dem = ee.Image('NASA/NASADEM_HGT/001') 
var elevation = dem.select('elevation').float();
//----------------------------------------------------KT---------------------------------------------------------
var tcw=image.expression(
       "B2*(0.1509)+B3*(0.1973)+B4*(0.3279)+B5*(0.3406)+B6*(-0.7112)+B7*(-0.4572)",
       {
           "B2": image.select(["blue"]),
           "B3": image.select(["green"]),
           "B4": image.select(["red"]),
            "B5": image.select(["nir"]),
            "B6": image.select(["swir1"]),
            "B7": image.select(["swir2"]),
         })

 //--------------------------------------------TCW----------------------------------------------

function TCW(img){
         var tcw=image.expression(
         "B2*(0.1509)+B3*(0.1973)+B4*(0.3279)+B5*(0.3406)+B6*(-0.7112)+B7*(-0.4572)",
         {
             "B2": image.select(["blue"]),
             "B3": image.select(["green"]),
             "B4": image.select(["red"]),
             "B5": image.select(["nir"]),
             "B6": image.select(["swir1"]),
             "B7": image.select(["swir2"]),
         })
         return tcw
 }
var tcw=TCW(image).rename('tcw');
 function TCB(img){
               var tcb = image.expression(
           "B2*(0.3037)+B3*(0.2793)+B4*(0.4743)+B5*(0.7243)+B6*(0.0840)+B7*(0.1863)",
        {
             "B2": image.select(["blue"]),
             "B3": image.select(["green"]),
            "B4": image.select(["red"]),
            "B5": image.select(["nir"]),
            "B6": image.select(["swir1"]),
            "B7": image.select(["swir2"]),
         })
         return tcb
 }
var tcb=TCB(image).rename('tcb');
 function TCG(img){
   var tcg = image.expression(
         "B2*(-0.2848)+B3*(-0.2435)+B4*(-0.5436)+B5*(0.5585)+B6*(0.5082)+B7*(-0.1800)",
        {
             "B2": image.select(["blue"]),
             "B3": image.select(["green"]),
             "B4": image.select(["red"]),
             "B5": image.select(["nir"]),
             "B6": image.select(["swir1"]),
             "B7": image.select(["swir2"]),
         });
         return tcg;
}
var tcg=TCG(image).rename('tcg');
         

    

//-------------------------------------------------Wet----------------------------------------------
 function WET(img){
 var wet = img.expression(
   'B*(0.1509) + G*(0.1973) + R*(0.3279) + NIR*(0.3406) + SWIR1*(-0.7112) + SWIR2*(-0.4572)',
   {
       'B': img.select('blue'),
       'G': img.select('green'),
       'R': img.select('red'),
       'NIR': img.select('nir'),
       'SWIR1': img.select('swir1'),
       'SWIR2': img.select('swir2')
   }
   );
   return wet;
 }

 var wet = WET(image).rename('wet');

 //--------------------------------------------LWSI----------------------------------------------
 function LWSI1(img){
          var lwsi1 = img.expression(
               "(NIR - SWIR1)/(NIR + SWIR1)",
      {
               'NIR':img.select('nir'),
               'SWIR1': img.select('swir1')
   }
   );
   return lwsi1;
 }

 var lwsi1 = LWSI1(image).rename('lwsi1');

 function LWSI2(img){
        var lwsi2 = img.expression(
          "(NIR - SWIR2)/(NIR + SWIR2)",
       {
          'NIR':img.select('nir'),
          'SWIR2': img.select('swir2')
   }
   );
   return lwsi2;
 }

 var lwsi2 = LWSI2(image).rename('lwsi2').clip(table);
//--------------------------------------------RVI----------------------------------------------
function EVI(img){
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
    {
      'NIR' : img.select('nir'),
      'RED' : img.select('red'),
      'BLUE': img.select('blue')
    }
  );
  return evi;
}

var evi_Vis = {
  min: 0.0,
  max: 1.0,
  palette: [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301'
  ],
};

var evi = EVI(image).rename('evi').clip(table1);




 //------------------------------------------第四层 偏干性常绿阔叶林、偏湿性常绿阔叶林-----------------------------------------
//----------------------------------------GLCM--------------------------------------------------



 //Random Forest
 //合成分类影像
var forest_mask=forest.eq(1)
var evergreen_mask=evergreen.eq(1)
var broadleaf_mask=ebf.eq(2);//掩膜建立的值根据上一步结果选择
var image = image.updateMask(forest_mask).updateMask(evergreen_mask).updateMask(broadleaf_mask)
                  .addBands(ndvi).addBands(evi).addBands(elevation).addBands(wet).addBands(tcw).addBands(lwsi1).addBands(lwsi2)/*.addBands(lst).addBands(vv).addBands(vh)*/;       //加啥波段自定
/*var textureimage=image4.select("VV","VH")
var input =textureimage.toUint16()
var glcm = input.glcmTexture();
print(glcm)*/

 // 将分类样本点合并在一起（根据Assets中的名称改写（常绿针叶林、落叶阔叶林））
var points = sample_dry.merge(sample_wet);
 //print('points4', points4);
//随机排列训练样本点
points = points.randomColumn('random');
points = points.sort('random');
// 选择训练波段（直接选择影像则将所有波段用于分类，也可以自行选择）
var bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2','slope','ndvi','evi','wet',/*'lst',*/'tcw','lwsi1','elevation',/*,'VV','VH'*/];

 // 对输入的数据集采样，生成一个训练样本集（注意properties和自己Assets中点属性名字一致）
var training = image.select(bands).sampleRegions({
   collection: points,
   properties: ['wet'],
   scale: 30,
   tileScale: 16
 });

// // 随机选取样本
/* var withRandom4 = training4.randomColumn('random');//样本点随机的排列

// // 我们想保留一些数据进行测试，以避免模型过度拟合。
 var split1 = 0.8; 
 var split2 = 0.2; 
 var trainingPartition4 = withRandom4.filter(ee.Filter.lt('random', split1));// 筛选80%的样本作为训练样本
 var testingPartition4 = withRandom4.filter(ee.Filter.gte('random', split2));// 筛选20%的样本作为测试样本

// // 选择分类的属性（注意和自己Assets中点属性名字一致）
 var classProperty4 = 'wet';

 // 分类方法选择randomForest(10这个值可以根据后面的调试代码修改)
 var classifier_Randomforest4 = ee.Classifier.smileRandomForest(25).train({
   features: trainingPartition4,
   classProperty: 'wet',
   inputProperties: bands4
 });
print(classifier_Randomforest4.explain());
// // 开始分类
 var classified_Randomforest4 = image4.select(bands4).classify(classifier_Randomforest4);

// // 运用测试样本分类，确定要进行函数运算的数据集以及函数
 var test_Randomforest4 = testingPartition4.classify(classifier_Randomforest4);

// // 计算混淆矩阵
 var confusionMatrix_Randomforest4 = test_Randomforest4.errorMatrix('wet', 'classification');
 print('Randomforest confusionMatrix', confusionMatrix_Randomforest4);// 面板上显示混淆矩阵
 print('Randomforest consumers accuracy', confusionMatrix_Randomforest4.consumersAccuracy());
 print('Randomforest producers accuracy', confusionMatrix_Randomforest4.producersAccuracy());
 print('Randomforest overall accuracy', confusionMatrix_Randomforest4.accuracy());// 面板上显示总体精度
 print('Randomforest kappa accuracy', confusionMatrix_Randomforest4.kappa());//面板上显示kappa值


 // 八邻域空间滤波处理，平滑影像
 var smooth_map_Randomforest4 = classified_Randomforest4
                     .focal_mode({
                      radius: 2, kernelType: 'octagon', units: 'pixels', iterations: 1
                    })
                     .mask(classified_Randomforest4.gte(1))

// // 空间连通性处理，去除小斑块
 var crude_object_removal_Randomforest4 = classified_Randomforest4
                               .updateMask(classified_Randomforest4.connectedPixelCount(2, false).gte(2))
                               .unmask(smooth_map_Randomforest4);


 // 定义土地利用分类数据的可视化参数（16进制颜色表中查找,注释中括号的值为分类的代码值，按大小排列的）
 var palette4 = [
  '#0ec613', // dry (1)  
   'ff0000', //  wet (2) 
 ];

 // 显示分类后影像
Map.addLayer(crude_object_removal_Randomforest4, {min: 1, max:2, palette: palette4}, 'Classification_Layer4');

//选取森林棵树（用于优化模型）
/*var numTrees4 = ee.List.sequence(5, 50, 5); 
var accuracies4 = numTrees4.map(function(t)
{ 
  var classifier4 = ee.Classifier.smileRandomForest(t)
                    .train({
                     features: trainingPartition4,
                     classProperty: 'wet',
                     inputProperties: bands4
                    });
  return testingPartition4
      .classify(classifier4)
      .errorMatrix('wet', 'classification')
      .accuracy();
}); 
print(ui.Chart.array.values({
  array: ee.Array(accuracies4),
  axis: 0,
  xLabels: numTrees4
}));

Export.image.toDrive({
  image:  crude_object_removal_Randomforest4,
  description: ' crude_object_removal_Randomforest4',
  scale: 30,
  region: table.geometry().bounds(),
  maxPixels: 1e13
});

*/



//----------------------------------------------------K折RF---------------------------------------------------------

var k = 10;
var number_points = training.size();
print('Number of points', number_points);
var fold_size = number_points.divide(k).floor();
print('Fold size', fold_size);

var epochs = ee.List.sequence(0, k-2);
print('Epochs number', epochs.size());
print('Epoch', epochs);

//定义土地利用分类数据的可视化参数（16进制颜色表中查找,注释中括号的值为分类的代码值，按大小排列的）
var palette = [
  '228B22',
  '#f00000'
];

epochs.evaluate(function(nums) {
  var j = 0;
  var OA = 0;
  var Kappa = 0;
  nums.map(function(epoch){
    var i = ee.Number(epoch);
    var list1 = training.toList(fold_size, i.multiply(fold_size));
    var testingPartition = ee.FeatureCollection(list1);
    var list2 = training.toList(number_points).removeAll(list1);
    var trainingPartition = ee.FeatureCollection(list2);
    //print(testingPartition);
    //print(trainingPartition);
      
    //分类方法选择randomForest(10这个值可以根据后面的调试代码修改，建议单独开代码调整)
    var classifier_Randomforest = ee.Classifier.smileRandomForest(30).train({
      features: trainingPartition,
      classProperty: 'wet',
      inputProperties: bands
    });
    
    //开始分类
    var classified_Randomforest = image.select(bands).classify(classifier_Randomforest);

    //运用测试样本分类，确定要进行函数运算的数据集以及函数
    var test_Randomforest = testingPartition.classify(classifier_Randomforest);
    //选取森林棵树（用于优化模型）

    //计算混淆矩阵
    var confusionMatrix_Randomforest = test_Randomforest.errorMatrix('wet', 'classification');
    print('第'+(j+1)+'折'+' Randomforest confusionMatrix', confusionMatrix_Randomforest);// 面板上显示混淆矩阵
    print('第'+(j+1)+'折'+' Randomforest consumers accuracy', confusionMatrix_Randomforest.consumersAccuracy());
     print('第'+(j+1)+'折'+' Randomforest producers accuracy', confusionMatrix_Randomforest.producersAccuracy());
    print('第'+(j+1)+'折'+' Randomforest overall accuracy', confusionMatrix_Randomforest.accuracy());// 面板上显示总体精度
    print('第'+(j+1)+'折'+' Randomforest kappa accuracy', confusionMatrix_Randomforest.kappa());//面板上显示kappa值
    
    //显示分类后影像(未平滑)
    Map.addLayer(classified_Randomforest, {min: 1, max:2, palette: palette}, '第'+(j+1)+'折'+' classification', false);
    
    OA = ee.Number(OA).add(ee.Number(confusionMatrix_Randomforest.accuracy()));
    Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_Randomforest.kappa()));
  var numTrees = ee.List.sequence(5, 100, 5); 
var accuracies = numTrees.map(function(t)
{ 
  var classifier = ee.Classifier.smileRandomForest(t)
                    .train({
                     features: trainingPartition,
                     classProperty: 'wet',
                     inputProperties: bands
                    });
  return testingPartition
      .classify(classifier)
      .errorMatrix('wet', 'classification')
      .accuracy();
}); 
print(ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTrees
}));
    //导入Asset(后续掩膜)，选择单折精度最高的结果，最后一折的导出语句在后面
    Export.image.toAsset({
      image: classified_Randomforest,
      description: '第'+(j+1)+'折' + ' RandomForest Forest_nonForest',
      assetId: 'wet',
      scale: 30,
      region: table.geometry().bounds(),
      maxPixels: 1e13
    });
    j++;
  })
  
  //最后一组无法整除，单独计算
  var last_testingPartition_size = ee.Number(number_points).subtract(ee.Number(k).subtract(1).multiply(fold_size));
  var last_trainingPartition_size = ee.Number(k).subtract(1).multiply(fold_size);

  var list1 = training.toList(last_testingPartition_size, last_trainingPartition_size);
  var testingPartition = ee.FeatureCollection(list1);
  var list2 = training.toList(number_points).removeAll(list1);
  var trainingPartition = ee.FeatureCollection(list2);
  //print(testingPartition);
  //print(trainingPartition);
  
  //(10这个值可以根据后面的调试代码修改，建议单独开代码调整)
  var classifier_Randomforest = ee.Classifier.smileRandomForest(25).train({
    features: trainingPartition,
    classProperty:'wet',
    inputProperties: bands
  });
    
  var classified_Randomforest = image.select(bands).classify(classifier_Randomforest);

  var test_Randomforest = testingPartition.classify(classifier_Randomforest);

  //计算混淆矩阵
  var confusionMatrix_Randomforest = test_Randomforest.errorMatrix('wet', 'classification');
  print('第'+k+'折'+' Randomforest confusionMatrix', confusionMatrix_Randomforest);// 面板上显示混淆矩阵
  print('第'+k+'折'+' Randomforest consumers accuracy', confusionMatrix_Randomforest.consumersAccuracy());
  print('第'+k+'折'+' Randomforest producers accuracy', confusionMatrix_Randomforest.producersAccuracy());
  print('第'+k+'折'+' Randomforest overall accuracy', confusionMatrix_Randomforest.accuracy());// 面板上显示总体精度
  print('第'+k+'折'+' Randomforest kappa accuracy', confusionMatrix_Randomforest.kappa());//面板上显示kappa值

  Map.addLayer(classified_Randomforest, {min: 1, max:2, palette: palette}, '第'+k+'折'+' classification', false);
  
  OA = ee.Number(OA).add(ee.Number(confusionMatrix_Randomforest.accuracy()))
  Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_Randomforest.kappa()))
  print('RandomForest Average OA', OA.divide(ee.Number(k)));
  print('RandomForest Average Kappa', Kappa.divide(ee.Number(k)));

  //导入Asset(后续掩膜)，选择单折精度最高的结果
  Export.image.toAsset({
    image: classified_Randomforest,
    description: '第'+k+'折' + ' RandomForest_bef',
    assetId: 'bef_collection',
    scale: 30,
    region: table.geometry().bounds(),
    maxPixels: 1e13
  });
})


//--------------------------------------------SVM(KFold=5)-----------------------------------------------
/*var k = 10;
var number_points = training.size();
print('Number of points', number_points);
var fold_size = number_points.divide(k).floor();
print('Fold size', fold_size);

var epochs = ee.List.sequence(0, k-2);
print('Epochs number', epochs.size());
print('Epoch', epochs);

//定义土地利用分类数据的可视化参数（16进制颜色表中查找,注释中括号的值为分类的代码值，按大小排列的）
var palette = [
  '228B22',
  'F0E68C',
];

epochs.evaluate(function(nums) {
  var j = 0;
  var OA = 0;
  var Kappa = 0;
  nums.map(function(epoch){
    var i = ee.Number(epoch);
    var list1 = training.toList(fold_size, i.multiply(fold_size));
    var testingPartition = ee.FeatureCollection(list1);
    var list2 = training.toList(number_points).removeAll(list1);
    var trainingPartition = ee.FeatureCollection(list2);
    //print(testingPartition);
    //print(trainingPartition);
      
    //分类方法选择libsvm
    var classifier_SVM = ee.Classifier.libsvm().train({
      features: trainingPartition,
      classProperty: 'wet',
      inputProperties: bands
    });

    //开始分类
    var classified_SVM = image.select(bands).classify(classifier_SVM);

    //运用测试样本分类，确定要进行函数运算的数据集以及函数
    var test_SVM = testingPartition.classify(classifier_SVM);

    //计算混淆矩阵
    var confusionMatrix_SVM = test_SVM.errorMatrix('wet', 'classification');
    print('第'+(j+1)+'折'+' SVM confusionMatrix', confusionMatrix_SVM);// 面板上显示混淆矩阵
    print('第'+(j+1)+'折'+' SVM consumers accuracy', confusionMatrix_SVM.consumersAccuracy());
    print('第'+(j+1)+'折'+' SVM producers accuracy', confusionMatrix_SVM.producersAccuracy());
    print('第'+(j+1)+'折'+' SVM overall accuracy', confusionMatrix_SVM.accuracy());// 面板上显示总体精度
    print('第'+(j+1)+'折'+' SVM kappa accuracy', confusionMatrix_SVM.kappa());//面板上显示kappa值
    
    //显示分类后影像(未平滑)
    Map.addLayer(classified_SVM, {min: 1, max:2, palette: palette}, '第'+(j+1)+'折'+' classification', false);

    OA = ee.Number(OA).add(ee.Number(confusionMatrix_SVM.accuracy()));
    Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_SVM.kappa()));
    
    //导入Asset(后续掩膜)，选择单折精度最高的结果，最后一折的导出语句在后面
    Export.image.toAsset({
      image: classified_SVM,
      description: '第'+(j+1)+'折' + ' SVM_evergreen_nonevergreen',
      assetId: 'evergreen_collection',
      scale: 30,
      region: table.geometry().bounds(),
      maxPixels: 1e13
    });
    j++;
  })
  
  //最后一组无法整除，单独计算
  var last_testingPartition_size = ee.Number(number_points).subtract(ee.Number(k).subtract(1).multiply(fold_size));
  var last_trainingPartition_size = ee.Number(k).subtract(1).multiply(fold_size);

  var list1 = training.toList(last_testingPartition_size, last_trainingPartition_size);
  var testingPartition = ee.FeatureCollection(list1);
  var list2 = training.toList(number_points).removeAll(list1);
  var trainingPartition = ee.FeatureCollection(list2);
  //print(testingPartition);
  //print(trainingPartition);

  var classifier_SVM = ee.Classifier.libsvm().train({
    features: trainingPartition,
    classProperty:'wet',
    inputProperties: bands
  });

  var classified_SVM = image.select(bands).classify(classifier_SVM);

  var test_SVM = testingPartition.classify(classifier_SVM);

  //计算混淆矩阵
  var confusionMatrix_SVM = test_SVM.errorMatrix('wet', 'classification');
  print('第'+k+'折'+' SVM confusionMatrix', confusionMatrix_SVM);// 面板上显示混淆矩阵
  print('第'+k+'折'+' SVM consumers accuracy', confusionMatrix_SVM.consumersAccuracy());
  print('第'+k+'折'+' SVM producers accuracy', confusionMatrix_SVM.producersAccuracy());
  print('第'+k+'折'+' SVM overall accuracy', confusionMatrix_SVM.accuracy());// 面板上显示总体精度
  print('第'+k+'折'+' SVM kappa accuracy', confusionMatrix_SVM.kappa());//面板上显示kappa值

  Map.addLayer(classified_SVM, {min: 1, max:2, palette: palette}, '第'+k+'折'+' classification', false);
  
  OA = ee.Number(OA).add(ee.Number(confusionMatrix_SVM.accuracy()))
  Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_SVM.kappa()))
  print('SVM Average OA', OA.divide(ee.Number(k)));
  print('SVM Average Kappa', Kappa.divide(ee.Number(k)));
  
  //导入Asset(后续掩膜)，选择单折精度最高的结果
  Export.image.toAsset({
    image: classified_SVM,
    description: '第'+k+'折' + ' SVM evergreen',
    assetId: 'evergreen_collection',
    scale: 30,
    region: table.geometry().bounds(),
    maxPixels: 1e13
  });
})*/






//---------------------------------------GradientTreeBoost(KFold=5)-------------------------------------------
/*var k =10;
var number_points = training.size();
print('Number of points', number_points);
var fold_size = number_points.divide(k).floor();
print('Fold size', fold_size);

var epochs = ee.List.sequence(0, k-2);
print('Epochs number', epochs.size());
print('Epoch', epochs);

//定义土地利用分类数据的可视化参数（16进制颜色表中查找,注释中括号的值为分类的代码值，按大小排列的）
var palette = [
  '228B22',
  '#f00000'
];

epochs.evaluate(function(nums) {
  var j = 0;
  var OA = 0;
  var Kappa = 0;
  nums.map(function(epoch){
    var i = ee.Number(epoch);
    var list1 = training.toList(fold_size, i.multiply(fold_size));
    var testingPartition = ee.FeatureCollection(list1);
    var list2 = training.toList(number_points).removeAll(list1);
    var trainingPartition = ee.FeatureCollection(list2);
    //print(testingPartition);
    //print(trainingPartition);
      
    //分类方法选择GradientTreeBoost(10这个值可以根据后面的调试代码修改，建议单独开代码调整)
    var classifier_GradientTreeBoost = ee.Classifier.smileGradientTreeBoost(55).train({
      features: trainingPartition,
      classProperty: 'wet',
      inputProperties: bands
    });

    //开始分类
    var classified_GradientTreeBoost = image.select(bands).classify(classifier_GradientTreeBoost);

    //运用测试样本分类，确定要进行函数运算的数据集以及函数
    var test_GradientTreeBoost = testingPartition.classify(classifier_GradientTreeBoost);
    
//选取森林棵树（用于优化模型）
var numTrees = ee.List.sequence(5, 100, 5); 
var accuracies = numTrees.map(function(t)
{ 
  var classifier = ee.Classifier.smileGradientTreeBoost(t)
                    .train({
                     features: trainingPartition,
                     classProperty: 'wet',
                     inputProperties: bands
                    });
  return testingPartition
      .classify(classifier)
      .errorMatrix('wet', 'classification')
      .accuracy();
}); 
print(ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTrees
}));


    //计算混淆矩阵
    var confusionMatrix_GradientTreeBoost = test_GradientTreeBoost.errorMatrix('wet', 'classification');
    print('第'+(j+1)+'折'+' GradientTreeBoost confusionMatrix', confusionMatrix_GradientTreeBoost);// 面板上显示混淆矩阵
    print('第'+(j+1)+'折'+' GradientTreeBoost consumers accuracy', confusionMatrix_GradientTreeBoost.consumersAccuracy());
    print('第'+(j+1)+'折'+' GradientTreeBoost producers accuracy', confusionMatrix_GradientTreeBoost.producersAccuracy());
    print('第'+(j+1)+'折'+' GradientTreeBoost overall accuracy', confusionMatrix_GradientTreeBoost.accuracy());// 面板上显示总体精度
    print('第'+(j+1)+'折'+' GradientTreeBoost kappa accuracy', confusionMatrix_GradientTreeBoost.kappa());//面板上显示kappa值
    
    //显示分类后影像(未平滑)
    Map.addLayer(classified_GradientTreeBoost, {min: 1, max:2, palette: palette}, '第'+(j+1)+'折'+' classification', false);

    OA = ee.Number(OA).add(ee.Number(confusionMatrix_GradientTreeBoost.accuracy()));
    Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_GradientTreeBoost.kappa()));
    
    //导入Asset(后续掩膜)，选择单折精度最高的结果，最后一折的导出语句在后面
    Export.image.toAsset({
      image: classified_GradientTreeBoost,
      description: '第'+(j+1)+'折' + ' GradientTreeBoost Forest_nonForest',
      assetId: 'Forest_collection',
      scale: 30,
      region: table.geometry().bounds(),
      maxPixels: 1e13
    });
    j++;
  })
  
  //最后一组无法整除，单独计算
  var last_testingPartition_size = ee.Number(number_points).subtract(ee.Number(k).subtract(1).multiply(fold_size));
  var last_trainingPartition_size = ee.Number(k).subtract(1).multiply(fold_size);

  var list1 = training.toList(last_testingPartition_size, last_trainingPartition_size);
  var testingPartition = ee.FeatureCollection(list1);
  var list2 = training.toList(number_points).removeAll(list1);
  var trainingPartition = ee.FeatureCollection(list2);
  //print(testingPartition);
  //print(trainingPartition);

  //(10这个值可以根据后面的调试代码修改，建议单独开代码调整)
  var classifier_GradientTreeBoost = ee.Classifier.smileGradientTreeBoost(55).train({
    features: trainingPartition,
    classProperty: 'wet',
    inputProperties: bands
  });

  var classified_GradientTreeBoost = image.select(bands).classify(classifier_GradientTreeBoost);

  var test_GradientTreeBoost = testingPartition.classify(classifier_GradientTreeBoost);

  //计算混淆矩阵
  var confusionMatrix_GradientTreeBoost = test_GradientTreeBoost.errorMatrix('wet', 'classification');
  print('第'+k+'折'+' GradientTreeBoost confusionMatrix', confusionMatrix_GradientTreeBoost);// 面板上显示混淆矩阵
  print('第'+k+'折'+' GradientTreeBoost consumers accuracy', confusionMatrix_GradientTreeBoost.consumersAccuracy());
  print('第'+k+'折'+' GradientTreeBoost producers accuracy', confusionMatrix_GradientTreeBoost.producersAccuracy());
  print('第'+k+'折'+' GradientTreeBoost overall accuracy', confusionMatrix_GradientTreeBoost.accuracy());// 面板上显示总体精度
  print('第'+k+'折'+' GradientTreeBoost kappa accuracy', confusionMatrix_GradientTreeBoost.kappa());//面板上显示kappa值

  Map.addLayer(classified_GradientTreeBoost, {min: 1, max:2, palette: palette}, '第'+k+'折'+' classification', false);
  
  OA = ee.Number(OA).add(ee.Number(confusionMatrix_GradientTreeBoost.accuracy()))
  Kappa = ee.Number(Kappa).add(ee.Number(confusionMatrix_GradientTreeBoost.kappa()))
  print('GradientTreeBoost Average OA', OA.divide(ee.Number(k)));
  print('GradientTreeBoost Average Kappa', Kappa.divide(ee.Number(k)));
  
  var numTrees = ee.List.sequence(5, 100, 5); 
var accuracies = numTrees.map(function(t)
{ 
  var classifier = ee.Classifier.smileGradientTreeBoost(t)
                    .train({
                     features: trainingPartition,
                     classProperty: 'wet',
                     inputProperties: bands
                    });
  return testingPartition
      .classify(classifier)
      .errorMatrix('wet', 'classification')
      .accuracy();
}); 
print(ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTrees
}));

  //导入Asset(后续掩膜)，选择单折精度最高的结果
  Export.image.toAsset({
    image: classified_GradientTreeBoost,
    description: '第'+k+'折' + ' GradientTreeBoost Forest_nonForest',
    assetId: 'Forest_collection',
    scale:30,
    region: table.geometry().bounds(),
    maxPixels: 1e13
  });
})
*/
