package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"github.com/petar/GoMNIST"
)

func printImage(image GoMNIST.RawImage) {
	scaleImage := 255.0 / 8.0
	numRow := 28
	numCol := 28

	backgroundChar := "█" // Use a solid gray/dark character

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

			// Print either space or solid gray background
			if scalePixel == 0 {
				fmt.Print(backgroundChar)
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}
}
func main() {
	rand.Seed(999)
	startTime := time.Now()

	train, test, err := GoMNIST.Load("../data")
	if err != nil {
		panic(err)
	}

	images := make([][]float64, len(train.Images))
	labels := make([]int, len(train.Images))

	for i := 0; i < len(train.Images); i++ {
		images[i] = make([]float64, len(train.Images[0]))
		for j := range train.Images[0] {
			images[i][j] = float64(train.Images[i][j])
			labels[i] = int(train.Labels[i])
		}
	}

	// Normalize pixel values to the range [0, 1]
	normalizeImages(train.Images)
	normalizeImages(test.Images)

	// Convert GO-MNIST data to training.Examples format
	trainExamples := convertToExamples(train.Images, convertLabelsToInt(train.Labels))
	testExamples := convertToExamples(test.Images, convertLabelsToInt(test.Labels))

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(trainExamples[0].Input),
		Layout:     []int{50, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1),
		Bias:       true,
	})

	trainer := training.NewBatchTrainer(training.NewAdam(0.02, 0.9, 0.999, 1e-8), 1, 200, 8)

	// Print the first image
	fmt.Println("First Train Label: ", train.Labels[0])
	printImage(train.Images[0])

	fmt.Printf("training: %d, val: %d, test: %d\n", len(trainExamples), len(testExamples), len(testExamples))

	trainer.Train(neural, trainExamples, testExamples, 500)

	duration := time.Since(startTime)
	fmt.Printf("Program duration: %s\n", duration)

}

func normalizeImages(imagesNormal []GoMNIST.RawImage) {
	for i := range imagesNormal {
		for j := range imagesNormal[i] {
			imagesNormal[i][j] = imagesNormal[i][j] / 255.0
		}
	}
}

func convertLabelsToInt(labelsNormal []GoMNIST.Label) []int {
	intLabelsNormal := make([]int, len(labelsNormal))
	for i, l := range labelsNormal {
		intLabelsNormal[i] = int(l)
	}
	return intLabelsNormal
}

func convertToExamples(imagesNormal []GoMNIST.RawImage, labels []int) training.Examples {
	var examples training.Examples
	for i := range imagesNormal {
		// Convert image bytes to []float64
		imageData := make([]float64, len(imagesNormal[i]))
		for j := range imagesNormal[i] {
			imageData[j] = float64(imagesNormal[i][j]) / 255.0
		}
		examples = append(examples, training.Example{
			Response: onehot(10, float64(labels[i])),
			Input:    imageData,
		})
	}
	return examples
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)] = 1
	return res
}
