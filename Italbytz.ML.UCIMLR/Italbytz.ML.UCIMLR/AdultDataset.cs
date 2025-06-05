using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class AdultDataset : Dataset<AdultDataset.AdultModelInput>
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Adult.csv";

    public override string FilePrefix { get; } = "adult";

    public override string? LabelColumnName { get; } = @"income";

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
            "ColumnName": "workclass",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "fnlwgt",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "education",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "education-num",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "marital-status",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "occupation",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "relationship",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "race",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "sex",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "capital-gain",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "capital-loss",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "hours-per-week",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "native-country",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "income",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]

        """;

    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair(@"relationship", @"relationship"),
                    new InputOutputColumnPair(@"race", @"race"),
                    new InputOutputColumnPair(@"sex", @"sex")
                })
            .Append(mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"age", @"age"),
                new InputOutputColumnPair(@"fnlwgt", @"fnlwgt"),
                new InputOutputColumnPair(@"education-num", @"education-num"),
                new InputOutputColumnPair(@"capital-gain", @"capital-gain"),
                new InputOutputColumnPair(@"capital-loss", @"capital-loss"),
                new InputOutputColumnPair(@"hours-per-week", @"hours-per-week")
            }))
            .Append(mlContext.Transforms.Text.FeaturizeText(
                inputColumnName: @"workclass", outputColumnName: @"workclass"))
            .Append(mlContext.Transforms.Text.FeaturizeText(
                inputColumnName: @"education", outputColumnName: @"education"))
            .Append(mlContext.Transforms.Text.FeaturizeText(
                inputColumnName: @"marital-status",
                outputColumnName: @"marital-status"))
            .Append(mlContext.Transforms.Text.FeaturizeText(
                inputColumnName: @"occupation",
                outputColumnName: @"occupation"))
            .Append(mlContext.Transforms.Text.FeaturizeText(
                inputColumnName: @"native-country",
                outputColumnName: @"native-country"))
            .Append(mlContext.Transforms.Concatenate(@"Features",
                @"relationship", @"race", @"sex", @"age", @"fnlwgt",
                @"education-num", @"capital-gain", @"capital-loss",
                @"hours-per-week", @"workclass", @"education",
                @"marital-status", @"occupation", @"native-country"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"income",
                @"income", addKeyValueAnnotationsAsText: false))
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
        return LoadFromTextFile<AdultModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    public class AdultModelInput
    {
        [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"workclass")]
        public string Workclass { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"fnlwgt")]
        public float Fnlwgt { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"education")]
        public string Education { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"education-num")]
        public float Education_num { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"marital-status")]
        public string Marital_status { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"occupation")]
        public string Occupation { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"relationship")]
        public string Relationship { get; set; }

        [LoadColumn(8)] [ColumnName(@"race")] public string Race { get; set; }

        [LoadColumn(9)] [ColumnName(@"sex")] public string Sex { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"capital-gain")]
        public float Capital_gain { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"capital-loss")]
        public float Capital_loss { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"hours-per-week")]
        public float Hours_per_week { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"native-country")]
        public string Native_country { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"income")]
        public string Income { get; set; }
    }
}