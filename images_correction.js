var table = table.geometry().bounds()
//-----------------------------------------------获取影像行列号-------------------------------------------------------
var collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")  
            .filterDate('2020-05-02', '2020-05-31')
            .filterBounds(table);

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
                    .filterDate('2020-05-1', '2020-09-30')            
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

// 将影像集合合成为单个影像
var mosaicImage = selected_images.mosaic();
var exportOptions = {
  image: mosaicImage.select(['SR_B4', 'SR_B3', 'SR_B2']),
  description: 'selected_images',
  scale: 500,                      // 设置导出的像素大小，这里假设为500米
  region: table,        // 设置导出区域，假设table是一个Feature或FeatureCollection
  fileFormat: 'GeoTIFF',           // 设置文件格式
  folder: 'image0127',             // 指定Google Drive上的文件夹名称
  maxPixels: 1e13                  // 设置最大像素限制
};

// 执行导出
Export.image.toDrive(exportOptions);


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
              .filterDate('2020-05-01', '2020-09-30')
              .filterBounds(table)
              .map(cloudfree_landsat)
              .median();

//三年影像填空值
var year_list = ee.List.sequence(2018,2021);
year_list = year_list.map(function(num){
  var start_time = ee.Date.fromYMD(num, 4, 1);
  var end_time = ee.Date.fromYMD(num, 10, 30);
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

// 将影像集合合成为单个影像
var mosaicImage = cloud_free_images.mosaic();

// 定义导出参数
var exportOptions = {
  image: mosaicImage.select(['SR_B4', 'SR_B3', 'SR_B2']),
  description: 'cloud_free_images',
  scale: 500,                      // 设置导出的像素大小，这里假设为500米
  region: table,        // 设置导出区域，假设table是一个Feature或FeatureCollection
  fileFormat: 'GeoTIFF',           // 设置文件格式
  folder: 'image0127',             // 指定Google Drive上的文件夹名称
  maxPixels: 1e13                  // 设置最大像素限制
};

// 执行导出
Export.image.toDrive(exportOptions);


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
print(images_correction)
var dataset=images_correction.clip(table)
Map.addLayer(table, {}, 'research_area', false);
Map.addLayer(images_collection, { bands: 'red,green,blue',min: 0.0, max: 0.3}, 'original', false);
Map.addLayer(dataset, { bands: 'red,green,blue',min: 0.0, max: 0.3}, 'corrected', false);

var forestPoints = ee.FeatureCollection('users/zhangshiqi/10sample_forest');
// 定义统计区域的函数
var getStats = function (image, region, scale) {
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(), // 或者您需要的其他统计方法
    geometry: region.geometry(),
    scale: scale
  });
  return ee.Feature(region).set(stats);
};

// 调用 getStats 函数
var scale = 30; // 例如，Landsat影像的空间分辨率为30米

// 获取校正前和校正后的统计信息
var statsBeforeCorrection = forestPoints.map(function (feature) {
  return getStats(images_collection.mean(), ee.Feature(feature), scale);
});

var statsAfterCorrection = forestPoints.map(function (feature) {
  return getStats(images_correction, ee.Feature(feature), scale);
});

// 转换成特征集合
var statsBeforeCorrectionCollection = ee.FeatureCollection(statsBeforeCorrection);
var statsAfterCorrectionCollection = ee.FeatureCollection(statsAfterCorrection);


// 导出为CSV文件
Export.table.toDrive({
  collection: statsBeforeCorrectionCollection,
  description: 'stats_before_correction', // 文件名前缀
  folder: 'your_folder_name', // 指定Google Drive上的文件夹名称
  fileFormat: 'CSV' // 导出格式为CSV
});

Export.table.toDrive({
  collection: statsAfterCorrectionCollection,
  description: 'stats_after_correction', // 文件名前缀
  folder: 'your_folder_name', // 指定Google Drive上的文件夹名称
  fileFormat: 'CSV' // 导出格式为CSV
});




// 定义导出参数
var exportOptions = {
  image: dataset.select(['red', 'green', 'blue']),
  description: 'dataset',
  scale: 500,                      // 设置导出的像素大小，这里假设为500米
  region: table,        // 设置导出区域，假设table是一个Feature或FeatureCollection
  fileFormat: 'GeoTIFF',           // 设置文件格式
  folder: 'image0127',             // 指定Google Drive上的文件夹名称
  maxPixels: 1e13                  // 设置最大像素限制
};

// 执行导出
Export.image.toDrive(exportOptions);
//----------------------------------------NDVI--------------------------------------------------
/*function NDVI(img){
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

var ndvi = NDVI(dataset).rename('ndvi');
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

var evi = EVI(dataset).rename('evi');
Map.addLayer(evi, evi_Vis, 'evi', false);
//----------------------------------------------------DEM---------------------------------------------------------
var dem = ee.Image('NASA/NASADEM_HGT/001') 
var elevation = dem.select('elevation').clip(table).float();
var image = dataset.addBands(ndvi).addBands(evi).addBands(elevation);
print(image)
Export.image.toDrive({
  image: image,
  description: 'simage',
  scale: 500,
  region: table,
  maxPixels: 1e13
});*/
