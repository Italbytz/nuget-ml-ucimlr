using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    CarEvaluationDataset : Dataset<CarEvaluationDataset.CarEvaluationModelInput>
{
    private readonly LookupMap<string>[] _lookupData =
    [
        new("unacc"),
        new("acc"),
        new("good"),
        new("vgood")
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.car_evaluation_strings.csv";

    public override string FilePrefix { get; } = "car_evaluation";

    public override string? LabelColumnName { get; } = @"class";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "buying",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "maint",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "doors",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "persons",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "lug_boot",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "safety",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "class",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "String",
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
        return LoadFromTextFile<CarEvaluationModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return mlContext.Transforms.ReplaceMissingValues(
            []);
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
        var buyingLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "vhigh" },
            new CategoryLookupMap { Value = 1f, Category = "high" },
            new CategoryLookupMap { Value = 2f, Category = "med" },
            new CategoryLookupMap { Value = 3f, Category = "low" }
        };
        var buyingLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(buyingLookupData);

        var maintLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "vhigh" },
            new CategoryLookupMap { Value = 1f, Category = "high" },
            new CategoryLookupMap { Value = 2f, Category = "med" },
            new CategoryLookupMap { Value = 3f, Category = "low" }
        };
        var maintLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(maintLookupData);

        var lugBootLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "small" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "big" }
        };
        var lugBootLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(lugBootLookupData);

        var safetyLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "low" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "high" }
        };
        var safetyLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(safetyLookupData);

        var doorsLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "two" },
            new CategoryLookupMap { Value = 1f, Category = "three" },
            new CategoryLookupMap { Value = 2f, Category = "four" },
            new CategoryLookupMap { Value = 3f, Category = "fiveormore" }
        };
        var doorsLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(doorsLookupData);

        var personsLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "two" },
            new CategoryLookupMap { Value = 1f, Category = "four" },
            new CategoryLookupMap { Value = 2f, Category = "more" }
        };
        var personsLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(personsLookupData);

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.Conversion.MapValue("buying",
                    buyingLookupIdvMap, buyingLookupIdvMap.Schema["Category"],
                    buyingLookupIdvMap.Schema["Value"], "buying").Append(
                    mlContext.Transforms.Conversion.MapValue(
                        @"maintenance", maintLookupIdvMap,
                        maintLookupIdvMap.Schema["Category"],
                        maintLookupIdvMap.Schema["Value"], "maint")).Append(
                    mlContext.Transforms.Conversion.MapValue(
                        @"lug_boot", lugBootLookupIdvMap,
                        lugBootLookupIdvMap.Schema["Category"],
                        lugBootLookupIdvMap.Schema["Value"], "lug_boot"))
                .Append(mlContext.Transforms.Conversion.MapValue(
                    @"safety", safetyLookupIdvMap,
                    safetyLookupIdvMap.Schema["Category"],
                    safetyLookupIdvMap.Schema["Value"], "safety"))
                .Append(mlContext.Transforms.Conversion.MapValue(
                    @"doors", doorsLookupIdvMap,
                    doorsLookupIdvMap.Schema["Category"],
                    doorsLookupIdvMap.Schema["Value"], "doors"))
                .Append(mlContext.Transforms.Conversion.MapValue(
                    @"persons", personsLookupIdvMap,
                    personsLookupIdvMap.Schema["Category"],
                    personsLookupIdvMap.Schema["Value"], "persons"))
                .Append(mlContext.Transforms.Concatenate(
                    @"Features", @"buying", @"maint", @"lug_boot", @"safety",
                    @"doors", @"persons"));
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Categorical.OneHotEncoding(
                    new[]
                    {
                        new InputOutputColumnPair(@"buying", @"buying"),
                        new InputOutputColumnPair(@"maint", @"maint"),
                        new InputOutputColumnPair(@"lug_boot", @"lug_boot"),
                        new InputOutputColumnPair(@"safety", @"safety"),
                        new InputOutputColumnPair(@"doors", @"doors"),
                        new InputOutputColumnPair(@"persons", @"persons")
                    })
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"buying",
                    @"maint", @"lug_boot", @"safety", @"doors", @"persons"));
        throw new NotImplementedException();
    }

    public class CarEvaluationModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"buying")]
        public string Buying { get; set; }

        [LoadColumn(1)] [ColumnName(@"maint")] public string Maint { get; set; }

        [LoadColumn(2)] [ColumnName(@"doors")] public string Doors { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"persons")]
        public string Persons { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"lug_boot")]
        public string Lug_boot { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"safety")]
        public string Safety { get; set; }

        [LoadColumn(6)] [ColumnName(@"class")] public string Class { get; set; }
    }
}