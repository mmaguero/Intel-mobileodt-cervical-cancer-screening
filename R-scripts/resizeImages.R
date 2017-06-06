# Clear workspace
rm(list=ls())


resizeImagesOfFolder <- function(read_dir_name, write_dir_name, img_quality) {
  
  img_file_list <- list.files(path = read_dir_name, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
  
  n_img <- length(img_file_list)
  
  for(i in 1:n_img) {
    img_file_name <- img_file_list[i]
    
    img <- readImage(img_file_name)
    # display(img, method="raster")
    img_resized <- resize(img, w=width, h=height)
    
    new_file_name <- gsub(read_dir_name, write_dir_name, img_file_name)
    
    j <- 1
    
    while(file.exists(new_file_name)){
      
      
      fileNameWOExt <- unlist(strsplit(new_file_name, "[.]"))[2]
      new_file_name<-sprintf(".%s_%i.jpg",fileNameWOExt,i)
      j<- j+1
      
    }
    print(sprintf("%s [%i of %i] write to: %s", read_dir_name, i, n_img, new_file_name))
    writeImage(img_resized, new_file_name, quality = img_quality)
  }
  
}


library(EBImage)

trainPath <- "./train"
testPath <- "./test"
type1ExtraPath <- "./Type_1"
type2ExtraPath <- "./Type_2"
type3ExtraPath <- "./Type_3"
write_dir_name <-"./train_256_extra"

width <- 256
height <- 256


# Load, resize and save test images
resizeImagesOfFolder(testPath, "./test_256", 85)

# Load, resize and save train images
resizeImagesOfFolder(trainPath, write_dir_name, 85)

# Load, resize and save type 1 extra images
resizeImagesOfFolder(type1ExtraPath, write_dir_name, 85)
# Load, resize and save type 2 extra images
resizeImagesOfFolder(type2ExtraPath, write_dir_name, 85)
# Load, resize and save type 3 extra images
resizeImagesOfFolder(type3ExtraPath, write_dir_name, 85)


