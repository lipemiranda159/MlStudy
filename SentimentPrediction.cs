using Microsoft.ML.Data;
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public string Category { get; set; }    
}