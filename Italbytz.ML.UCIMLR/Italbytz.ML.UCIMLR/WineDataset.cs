using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class WineDataset : Dataset<WineDataset.WineModelInput>
{
    private readonly LookupMap<float>[] _lookupData =
    [
        new(1.0f),
        new(2.0f),
        new(3.0f)
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Wine.csv";

    public override string FilePrefix { get; } = "wine";

    public override string? LabelColumnName { get; } = @"class";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "Alcohol",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Malicacid",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Ash",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Alcalinity_of_ash",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Magnesium",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Total_phenols",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Flavanoids",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Nonflavanoid_phenols",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Proanthocyanins",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Color_intensity",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Hue",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "0D280_0D315_of_diluted_wines",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Proline",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "class",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;

    public override IDataView LoadFromTextFile(string path,
        char? separatorChar = null,
        bool? hasHeader = null, bool? allowQuoting = null,
        bool? trimWhitespace = null, bool? allowSparse = null)
    {
        return LoadFromTextFile<WineModelInput>(path, separatorChar, hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return processingType switch
        {
            ProcessingType.Standard => mlContext.Transforms.Conversion
                .MapValueToKey(@"class", @"class",
                    addKeyValueAnnotationsAsText: false)
                .Append(mlContext.Transforms.CopyColumns("Label", "class")),
            ProcessingType.FeatureBinningAndCustomLabelMapping => mlContext
                .Transforms.Conversion.MapValueToKey(@"Label", @"class",
                    keyData: mlContext.Data.LoadFromEnumerable(_lookupData)),
            _ => throw new NotImplementedException()
        };
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Concatenate(@"Features",
                @"Alcohol", @"Malicacid", @"Ash", @"Alcalinity_of_ash",
                @"Magnesium", @"Total_phenols", @"Flavanoids",
                @"Nonflavanoid_phenols", @"Proanthocyanins", @"Color_intensity",
                @"Hue", @"0D280_0D315_of_diluted_wines", @"Proline");

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.NormalizeBinning(new[]
                {
                    new InputOutputColumnPair(@"Alcohol", @"Alcohol"),
                    new InputOutputColumnPair(@"Malicacid", @"Malicacid"),
                    new InputOutputColumnPair(@"Ash", @"Ash"),
                    new InputOutputColumnPair(@"Alcalinity_of_ash",
                        @"Alcalinity_of_ash"),
                    new InputOutputColumnPair(@"Magnesium", @"Magnesium"),
                    new InputOutputColumnPair(@"Total_phenols",
                        @"Total_phenols"),
                    new InputOutputColumnPair(@"Flavanoids", @"Flavanoids"),
                    new InputOutputColumnPair(@"Nonflavanoid_phenols",
                        @"Nonflavanoid_phenols"),
                    new InputOutputColumnPair(@"Proanthocyanins",
                        @"Proanthocyanins"),
                    new InputOutputColumnPair(@"Color_intensity",
                        @"Color_intensity"),
                    new InputOutputColumnPair(@"Hue", @"Hue"),
                    new InputOutputColumnPair(@"0D280_0D315_of_diluted_wines",
                        @"0D280_0D315_of_diluted_wines"),
                    new InputOutputColumnPair(@"Proline", @"Proline")
                }, maximumBinCount: 4)
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"Alcohol", @"Malicacid", @"Ash", @"Alcalinity_of_ash",
                    @"Magnesium", @"Total_phenols", @"Flavanoids",
                    @"Nonflavanoid_phenols", @"Proanthocyanins",
                    @"Color_intensity", @"Hue", @"0D280_0D315_of_diluted_wines",
                    @"Proline"));
        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
            return mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"Alcohol", @"Alcohol"),
                new InputOutputColumnPair(@"Malicacid", @"Malicacid"),
                new InputOutputColumnPair(@"Ash", @"Ash"),
                new InputOutputColumnPair(@"Alcalinity_of_ash",
                    @"Alcalinity_of_ash"),
                new InputOutputColumnPair(@"Magnesium", @"Magnesium"),
                new InputOutputColumnPair(@"Total_phenols",
                    @"Total_phenols"),
                new InputOutputColumnPair(@"Flavanoids", @"Flavanoids"),
                new InputOutputColumnPair(@"Nonflavanoid_phenols",
                    @"Nonflavanoid_phenols"),
                new InputOutputColumnPair(@"Proanthocyanins",
                    @"Proanthocyanins"),
                new InputOutputColumnPair(@"Color_intensity",
                    @"Color_intensity"),
                new InputOutputColumnPair(@"Hue", @"Hue"),
                new InputOutputColumnPair(@"0D280_0D315_of_diluted_wines",
                    @"0D280_0D315_of_diluted_wines"),
                new InputOutputColumnPair(@"Proline", @"Proline")
            });

        throw new NotSupportedException(
            $"The scenario type {scenarioType} is not supported.");
    }


    /// <summary>
    ///     Represents the input data schema for the Iris dataset used in ML.NET
    ///     models.
    /// </summary>
    public class WineModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"Alcohol")]
        public float Alcohol { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"Malicacid")]
        public float Malicacid { get; set; }

        [LoadColumn(2)] [ColumnName(@"Ash")] public float Ash { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"Alcalinity_of_ash")]
        public float Alcalinity_of_ash { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"Magnesium")]
        public float Magnesium { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"Total_phenols")]
        public float Total_phenols { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"Flavanoids")]
        public float Flavanoids { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"Nonflavanoid_phenols")]
        public float Nonflavanoid_phenols { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"Proanthocyanins")]
        public float Proanthocyanins { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"Color_intensity")]
        public float Color_intensity { get; set; }

        [LoadColumn(10)] [ColumnName(@"Hue")] public float Hue { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"0D280_0D315_of_diluted_wines")]
        public float _0D280_0D315_of_diluted_wines { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"Proline")]
        public float Proline { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"class")]
        public float Class { get; set; }
    }
}