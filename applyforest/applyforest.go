package main

import (
	"flag"
	"fmt"
	"github.com/IlyaLab/CloudForest"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest.")
	predfn := flag.String("preds",
		"", "The name of a file to write the predictions into.")
	votefn := flag.String("votes",
		"", "The name of a file to write categorical vote totals to.")
	var num bool
	flag.BoolVar(&num, "mean", false, "Force numeric (mean) voting.")
	var cat bool
	flag.BoolVar(&cat, "mode", false, "Force categorical (mode) voting.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestreader := CloudForest.NewForestReader(forestfile)
	forest, err := forestreader.ReadForest()
	if err != nil {
		log.Fatal(err)
	}

	var predfile *os.File
	if *predfn != "" {
		predfile, err = os.Create(*predfn)
		if err != nil {
			log.Fatal(err)
		}
		defer predfile.Close()
	}

	var bb CloudForest.VoteTallyer
	if !cat && (num || strings.HasPrefix(forest.Target, "N")) {
		bb = CloudForest.NewNumBallotBox(data.Data[0].Length())
	} else {
		bb = CloudForest.NewCatBallotBox(data.Data[0].Length())

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		fmt.Printf("Target is %v in feature %v\n", forest.Target, targeti)
		er := bb.TallyError(data.Data[targeti])
		fmt.Printf("Error: %v\n", er)
	}
	if *predfn != "" {
		fmt.Printf("Outputting label predicted actual tsv to %v\n", *predfn)
		for i, l := range data.CaseLabels {
			actual := "NA"
			if hasTarget {
				actual = data.Data[targeti].GetStr(i)
			}
			fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, bb.Tally(i), actual)
		}
	}

	//Not thread safe code!
	if *votefn != "" {
		fmt.Printf("Outputting vote totals to %v\n", *votefn)
		cbb := bb.(*CloudForest.CatBallotBox)
		votefile, err := os.Create(*votefn)
		if err != nil {
			log.Fatal(err)
		}
		defer votefile.Close()
		fmt.Fprintf(votefile, ".")

		for _, lable := range cbb.CatMap.Back {
			fmt.Fprintf(votefile, "\t%v", lable)
		}
		fmt.Fprintf(votefile, "\n")

		for i, box := range cbb.Box {
			fmt.Fprintf(votefile, "%v", data.CaseLabels[i])

			for j, _ := range cbb.CatMap.Back {
				total := 0.0
				total = box.Map[j]

				fmt.Fprintf(votefile, "\t%v", total)

			}
			fmt.Fprintf(votefile, "\n")

		}
	}
}
