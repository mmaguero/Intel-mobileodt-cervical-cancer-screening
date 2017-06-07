##############################################################################
## resizeImages.R
## Script to read all the images (training, testing and aditionals) of the 
## competition, resize it and save it on disk to make it handleable.
##
## You need to create the following empty directories to make it work:
##  - test_256
##  - train_256_extra
##    - Type_1
##    - Type_2
##    - Type_3
##############################################################################

# Clear workspace
rm(list=ls())

# Function to read the images of a given directory, resize it and save it to 
# other directory with the given quality
resizeImagesOfFolder <- function(read_dir_name, write_dir_name, img_quality) {
  
  img_file_list <- list.files(path = read_dir_name, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
  
  n_img <- length(img_file_list)
  
  for(i in 1:n_img) {
    img_file_name <- img_file_list[i]
    
    
    img <- try(readImage(img_file_name),silent = TRUE)
    
    if(is.Image(img)){
      # display(img, method="raster")
      
      img_resized <- resize(img, w=width, h=height)
    }
    
    
    new_file_name <- gsub(read_dir_name, write_dir_name, img_file_name)
    
    j <- 1
    
    while(file.exists(new_file_name)){
      
      
      fileNameWOExt <- unlist(strsplit(new_file_name, "[.]"))[2]
      new_file_name<-sprintf(".%s_%i.jpg",fileNameWOExt,j)
      j<- j+1
      
    }
    print(sprintf("%s [%i of %i] write to: %s", img_file_name, i, n_img, new_file_name))
    try(writeImage(img_resized, new_file_name, quality = img_quality),silent = TRUE)
  }
  
}


library(EBImage)

trainPath <- "../data/train"
testPath <- "../data/test"
type1ExtraPath <- "../data/Type_1"
type2ExtraPath <- "../data/Type_2"
type3ExtraPath <- "../data/Type_3"
write_dir_name <-"../data/train_256_OK"

width <- 256
height <- 256

testDir = "../data/testDir"
dir.create(testDir
           )

# Load, resize and save test images
resizeImagesOfFolder(testPath, "./test_256", 85)

# Load, resize and save train images
resizeImagesOfFolder(trainPath, write_dir_name, 85)



# Load, resize and save type 1 extra images
resizeImagesOfFolder(type1ExtraPath, paste(write_dir_name,"/Type_1", sep = "" ), 85)
# Load, resize and save type 2 extra images
resizeImagesOfFolder(type2ExtraPath, paste(write_dir_name,"/Type_2", sep = "" ), 85)
# Load, resize and save type 3 extra images
resizeImagesOfFolder(type3ExtraPath, paste(write_dir_name,"/Type_3", sep = "" ), 85)


