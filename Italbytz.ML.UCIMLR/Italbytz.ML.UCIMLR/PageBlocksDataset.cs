using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class PageBlocksDataset : Dataset<PageBlocksDataset.PageBlocksModelInput>
{
    private readonly LookupMap<float>[] _lookupData =
    [
        new(1.0f),
        new(2.0f),
        new(3.0f),
        new(4.0f),
        new(5.0f)
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Page_Blocks_Classification.csv";

    public override string FilePrefix { get; } = "page_blocks_classification";

    public override string? LabelColumnName { get; } = @"class";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "height",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "length",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "area",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "eccen",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "p_black",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "p_and",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "mean_tr",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "blackpix",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "blackand",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "wb_trans",
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
        return LoadFromTextFile<PageBlocksModelInput>(path, separatorChar,
            hasHeader,
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
                @"height", @"length", @"area", @"eccen", @"p_black", @"p_and",
                @"mean_tr", @"blackpix", @"blackand", @"wb_trans");

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.NormalizeBinning(new[]
                {
                    new InputOutputColumnPair(@"height", @"height"),
                    new InputOutputColumnPair(@"length", @"length"),
                    new InputOutputColumnPair(@"area", @"area"),
                    new InputOutputColumnPair(@"eccen", @"eccen"),
                    new InputOutputColumnPair(@"p_black", @"p_black"),
                    new InputOutputColumnPair(@"p_and", @"p_and"),
                    new InputOutputColumnPair(@"mean_tr", @"mean_tr"),
                    new InputOutputColumnPair(@"blackpix", @"blackpix"),
                    new InputOutputColumnPair(@"blackand", @"blackand"),
                    new InputOutputColumnPair(@"wb_trans", @"wb_trans")
                }, maximumBinCount: 4)
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"height", @"length", @"area", @"eccen", @"p_black",
                    @"p_and", @"mean_tr", @"blackpix", @"blackand",
                    @"wb_trans"));
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
                new InputOutputColumnPair(@"height", @"height"),
                new InputOutputColumnPair(@"length", @"length"),
                new InputOutputColumnPair(@"area", @"area"),
                new InputOutputColumnPair(@"eccen", @"eccen"),
                new InputOutputColumnPair(@"p_black", @"p_black"),
                new InputOutputColumnPair(@"p_and", @"p_and"),
                new InputOutputColumnPair(@"mean_tr", @"mean_tr"),
                new InputOutputColumnPair(@"blackpix", @"blackpix"),
                new InputOutputColumnPair(@"blackand", @"blackand"),
                new InputOutputColumnPair(@"wb_trans", @"wb_trans")
            });

        throw new NotSupportedException(
            $"The scenario type {scenarioType} is not supported.");
    }


    /// <summary>
    ///     Represents the input data schema for the Iris dataset used in ML.NET
    ///     models.
    /// </summary>
    public class PageBlocksModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"height")]
        public float Height { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"length")]
        public float Length { get; set; }

        [LoadColumn(2)] [ColumnName(@"area")] public float Area { get; set; }

        [LoadColumn(3)] [ColumnName(@"eccen")] public float Eccen { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"p_black")]
        public float P_black { get; set; }

        [LoadColumn(5)] [ColumnName(@"p_and")] public float P_and { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"mean_tr")]
        public float Mean_tr { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"blackpix")]
        public float Blackpix { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"blackand")]
        public float Blackand { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"wb_trans")]
        public float Wb_trans { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"class")]
        public float Class { get; set; }
    }
}