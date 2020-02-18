using Microsoft.ML.Data;
public class Sentence
{
    [LoadColumn(0)]
    public int ID { get; set; }
    [LoadColumn(1)]
    public string Text { get; set; }
    [LoadColumn(2)]
    public string Category { get; set; }
}