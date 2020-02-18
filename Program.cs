using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace TestMl
{
    class Program
    {
        private static MLContext _mlContext;
        private static PredictionEngine<Sentence, SentimentPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);       
            var list = new List<Sentence>();

            list.Add(new Sentence()
            {
                ID = 1,
                Text = "Você é legal",
                Category = "elogio"

            });
            list.Add(new Sentence()
            {
                ID = 2,
                Text = "Você é chata",
                Category = "xingamento"

            });

            list.Add(new Sentence()
            {
                ID = 3,
                Text = "Bom dia",
                Category = "comprimento"
            });
            _trainingDataView = _mlContext.Data.LoadFromEnumerable<Sentence>(list);

            var pipeline = processData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            var sentence = new Sentence()
            {
                Text = "legal"
            };

            var prediction = _predEngine.Predict(sentence);
            Console.WriteLine(prediction.Category);
            Console.Read();
            

        }

        private static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Sentence, SentimentPrediction>(_trainedModel);
            return trainingPipeline;
        }

        private static IEstimator<ITransformer> processData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
            .Append(_mlContext.Transforms.Text.FeaturizeText("TextFeaturized","Text"))
            .Append(_mlContext.Transforms.Concatenate("Features", "TextFeaturized"))
            .AppendCacheCheckpoint(_mlContext);
            return pipeline;
        }
    }
}
