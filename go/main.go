package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"github.com/petar/GoMNIST"
	//grayscale "github.com/ddecoen/cnn-week10/pkgs"
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
	// Get the absolute path of the directory
	executablePath, err := os.Executable()
	if err != nil {
		fmt.Println("Error getting executable path:", err)
		return
	}

	// calculate the directory containing the executable
	executableDir := filepath.Dir(executablePath)

	rand.Seed(999)
	startTime := time.Now()

	// Construct the absolute path to the data directory
	dataDir := filepath.Join(executableDir, "../data")

	train, test, err := GoMNIST.Load(dataDir)
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
		Layout:     []int{32, 64, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1),
		Bias:       true,
	})

	trainer := training.NewBatchTrainer(training.NewAdam(0.005, 0.9, 0.999, 1e-8), 1, 200, 8)

	// Print the first image
	fmt.Println("First Train label: ", train.Labels[0])
	printImage(train.Images[0])

	fmt.Printf("training: %d, val: %d, test: %d\n", len(trainExamples), len(testExamples), len(testExamples))

	trainer.Train(neural, trainExamples, testExamples, 500)

	// Calculate accuracy on test examples
	testAccuracy := calculateAccuracy(neural, testExamples)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAccuracy*100)

	// Save accuracy to a CSV file
	accuracyData := [][]string{
		{"Test Accuracy", fmt.Sprintf("%.2f%%", testAccuracy*100)},
	}

	file, err := os.Create("accuracy.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, row := range accuracyData {
		err := writer.Write(row)
		if err != nil {
			panic(err)
		}
	}

	fmt.Println("Accuracy data saved to accuracy.csv")

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

func calculateAccuracy(neural *deep.Neural, examples training.Examples) float64 {
	correctPredictions := 0

	for _, example := range examples {
		predicted := neural.Predict(example.Input)
		predictedLabel := maxIndex(predicted)
		actualLabel := maxIndex(example.Response)

		if predictedLabel == actualLabel {
			correctPredictions++
		}
	}

	return float64(correctPredictions) / float64(len(examples))
}

func maxIndex(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]

	for i, val := range arr {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}
