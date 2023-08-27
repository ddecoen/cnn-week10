package main

import (
	"fmt"
	"math"
	"math/rand"

	//"os"
	//"path/filepath"
	"time"

	"github.com/petar/GoMNIST"
)

//This is will leverage GoMNIST package and print the images to the console
func printImageGS(image GoMNIST.RawImage) {
	scaleImage := 255.0 / 8.0
	numRow := 28
	numCol := 28

	// ASCII characters representing different levels of gray
	//grayChars := []string{" ", "░", "▒", "▓", "█", "▓", "▒", "░", " ", " "}

	for i := 0; i < numRow; i++ {
		for j := 0; j < numCol; j++ {
			// Get the pixel value
			pixel := image[i*numCol+j]

			// Scale the pixel
			scalePixel := int(math.Round(float64(pixel) / scaleImage))
			//scalePixel := int(pixel * 9 / 255)

			// Build in check to ensure that 0 scales correctly
			if pixel != 0 && scalePixel == 0 {
				scalePixel = 1
			}
			// Print either space or scaled pixel value
			if scalePixel == 0 {
				fmt.Print("█") // print the dot or grayscale values greater than 0
			} else {
				fmt.Print(" ")
			}
			// Print the corresponding grey character
			//fmt.Print(grayChars[scalePixel])
		}
		// Put in new line for each row
		fmt.Println()
	}
}

func main() {
	// Get the absolute path of the directory
	//executablePath, err := os.Executable()
	//if err != nil {
	//	fmt.Println("Error getting executable path:", err)
	//	return
	//}

	// calculate the directory containing the executable
	// executableDir := filepath.Dir(executablePath)

	// Set a fixed random seed for reproducibility
	rand.Seed(999) // You can use any integer value as the seed
	startTime := time.Now()

	// Construct the absolute path to the data directory
	//dataDir := filepath.Join(executableDir, "../data")

	// Read in the data and set up variables for images and labels
	train, _, _ := GoMNIST.Load("../../data")
	images := make([][]float64, len(train.Images))
	labels := make([]int, len(train.Images))

	for i := 0; i < len(train.Images); i++ {
		images[i] = make([]float64, len(train.Images[0]))
		for j := range train.Images[0] {
			images[i][j] = float64(train.Images[i][j])
			labels[i] = int(train.Labels[i])
		}
	}

	// describe the dataset
	numInstances := len(images)
	numFeatures := len(images[0])
	fmt.Printf("Data shape: %d instances x %d features\n", numInstances, numFeatures)

	fmt.Println("First Train Label: ", train.Labels[0])
	printImageGS(train.Images[0])

	duration := time.Since(startTime)
	fmt.Printf("Program duration: %s\n", duration)

}
