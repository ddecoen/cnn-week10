package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/e-XpertSolutions/go-iforest/v2/iforest"
	"github.com/petar/GoMNIST"
)

func main() {
	// Get the absolute path of the directory
	executablePath, err := os.Executable()
	if err != nil {
		fmt.Println("Error getting executable path:", err)
		return
	}

	// calculate the directory containing the executable
	executableDir := filepath.Dir(executablePath)

	// Set a fixed random seed for reproducibility
	rand.Seed(999) // You can use any integer value as the seed
	startTime := time.Now()

	// Construct the absolute path to the data directory
	dataDir := filepath.Join(executableDir, "../data")

	// Read in the data and set up variables for images and labels
	train, _, _ := GoMNIST.Load(dataDir)
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

	// set up parameters for iforest
	trees := 1000
	samples := 256
	threshold := 0.001 //can update this based on expectations

	forest := iforest.NewForest(trees, samples, threshold)

	// setting up training and testing
	fmt.Println("Starting training....")
	forest.Train(images)
	fmt.Println("Starting testing...")
	forest.Test(images)

	fmt.Println("Completed training and testing")

	// print out the outliers
	numOutliers := 0
	outlierIndices := []int{}
	for i, score := range forest.AnomalyScores {
		if score > threshold {
			numOutliers++
			outlierIndices = append(outlierIndices, i)
		}
	}
	fmt.Printf("Number of outliers: %d\n", numOutliers)

	// print out the indices of the outliers
	// commented this out as it was taking up too much screen space
	//fmt.Println("Indices of outliers:")
	//for _, idx := range outlierIndices {
	//	fmt.Println(idx)
	//}

	// write outliers to a json file
	outlierIndicesData := struct {
		OutlierIndices []int `json:"outlier_indices"`
	}{
		OutlierIndices: outlierIndices,
	}
	jsonData, err := json.MarshalIndent(outlierIndicesData, "", " ")
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return
	}
	jsonFilename := "outlier_indices.json"
	jsonFile, err := os.Create(jsonFilename)
	if err != nil {
		fmt.Println("Error creating JSON file:", err)
		return
	}
	defer jsonFile.Close()

	_, err = jsonFile.Write(jsonData)
	if err != nil {
		fmt.Println("Error writing JSON data:", err)
		return
	}

	fmt.Printf("Outlier indices saved to %s\n", jsonFilename)

	// write results to csv
	file, _ := os.Create("anomaly_scores_go.csv")
	write := csv.NewWriter(file)
	anomalyScores := make([][]string, len(forest.AnomalyScores)+1)
	for i := 0; i < len(anomalyScores); i++ {
		anomalyScores[i] = make([]string, 2)
		if i == 0 {
			anomalyScores[0][0] = "id"
			anomalyScores[0][1] = "anomalyScores"
		}
		if i != 0 {
			score := 0.5 - forest.AnomalyScores[i-1]
			anomalyScores[i][0] = fmt.Sprintf("%d", i-1)
			anomalyScores[i][1] = fmt.Sprintf("%f", score)
		}
	}

	write.WriteAll(anomalyScores)
	duration := time.Since(startTime)
	fmt.Printf("Program duration: %s\n", duration)
}
