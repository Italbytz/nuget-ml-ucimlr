using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    HeartDiseaseBinaryDataset : Dataset<
    HeartDiseaseBinaryDataset.HeartDiseaseModelInput>
{
    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "age",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "sex",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "cp",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "trestbps",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "chol",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "fbs",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "restecg",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "thalach",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "exang",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "oldpeak",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "slope",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "ca",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "thal",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "num",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Heart_Disease_Binary.csv";

    public override string FilePrefix { get; } = "heart_disease_binary";

    public override string? LabelColumnName { get; } = @"num";

    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"age", @"age"),
                new InputOutputColumnPair(@"sex", @"sex"),
                new InputOutputColumnPair(@"cp", @"cp"),
                new InputOutputColumnPair(@"trestbps", @"trestbps"),
                new InputOutputColumnPair(@"chol", @"chol"),
                new InputOutputColumnPair(@"fbs", @"fbs"),
                new InputOutputColumnPair(@"restecg", @"restecg"),
                new InputOutputColumnPair(@"thalach", @"thalach"),
                new InputOutputColumnPair(@"exang", @"exang"),
                new InputOutputColumnPair(@"oldpeak", @"oldpeak"),
                new InputOutputColumnPair(@"slope", @"slope"),
                new InputOutputColumnPair(@"ca", @"ca"),
                new InputOutputColumnPair(@"thal", @"thal")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"age",
                @"sex", @"cp", @"trestbps", @"chol", @"fbs", @"restecg",
                @"thalach", @"exang", @"oldpeak", @"slope", @"ca", @"thal"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"num",
                @"num", addKeyValueAnnotationsAsText: false))
            .Append(estimator)
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel",
                    @"PredictedLabel"));

        return pipeline;
    }

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<HeartDiseaseModelInput>(path, separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }


    public class HeartDiseaseModelInput
    {
        [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

        [LoadColumn(1)] [ColumnName(@"sex")] public float Sex { get; set; }

        [LoadColumn(2)] [ColumnName(@"cp")] public float Cp { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"trestbps")]
        public float Trestbps { get; set; }

        [LoadColumn(4)] [ColumnName(@"chol")] public float Chol { get; set; }

        [LoadColumn(5)] [ColumnName(@"fbs")] public float Fbs { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"restecg")]
        public float Restecg { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"thalach")]
        public float Thalach { get; set; }

        [LoadColumn(8)] [ColumnName(@"exang")] public float Exang { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"oldpeak")]
        public float Oldpeak { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"slope")]
        public float Slope { get; set; }

        [LoadColumn(11)] [ColumnName(@"ca")] public float Ca { get; set; }

        [LoadColumn(12)] [ColumnName(@"thal")] public float Thal { get; set; }

        [LoadColumn(13)] [ColumnName(@"num")] public float Num { get; set; }
    }
}