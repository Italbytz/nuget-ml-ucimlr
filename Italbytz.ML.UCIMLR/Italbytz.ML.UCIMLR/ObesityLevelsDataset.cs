using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    ObesityLevelsDataset : Dataset<ObesityLevelsDataset.ObesityLevelsModelInput>
{
    private readonly LookupMap<string>[] _lookupData =
    [
        new("Insufficient_Weight"),
        new("Normal_Weight"),
        new("Overweight_Level_I"),
        new("Overweight_Level_II"),
        new("Obesity_Type_I"),
        new("Obesity_Type_II"),
        new("Obesity_Type_III")
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Estimation_of_Obesity_Levels_Based_On_Eating_Habits_and_Physical_Condition.csv";

    public override string FilePrefix { get; } = "ol";

    public override string? LabelColumnName { get; } =
        @"NObeyesdad";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "Gender",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Age",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Height",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Weight",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "family_history_with_overweight",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Boolean",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "FAVC",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Boolean",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "FCVC",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "NCP",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "CAEC",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "SMOKE",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Boolean",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "CH2O",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "SCC",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Boolean",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "FAF",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "TUE",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "CALC",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "MTRANS",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "NObeyesdad",
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
        return LoadFromTextFile<ObesityLevelsModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair(@"Gender", @"Gender"),
                    new InputOutputColumnPair(@"family_history_with_overweight",
                        @"family_history_with_overweight"),
                    new InputOutputColumnPair(@"FAVC", @"FAVC"),
                    new InputOutputColumnPair(@"CAEC", @"CAEC"),
                    new InputOutputColumnPair(@"SMOKE", @"SMOKE"),
                    new InputOutputColumnPair(@"SCC", @"SCC"),
                    new InputOutputColumnPair(@"CALC", @"CALC"),
                    new InputOutputColumnPair(@"MTRANS", @"MTRANS")
                })
            .Append(mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"Age", @"Age"),
                new InputOutputColumnPair(@"Height", @"Height"),
                new InputOutputColumnPair(@"Weight", @"Weight"),
                new InputOutputColumnPair(@"FCVC", @"FCVC"),
                new InputOutputColumnPair(@"NCP", @"NCP"),
                new InputOutputColumnPair(@"CH2O", @"CH2O"),
                new InputOutputColumnPair(@"FAF", @"FAF"),
                new InputOutputColumnPair(@"TUE", @"TUE")
            }));
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.NormalizeBinning(new[]
            {
                new InputOutputColumnPair(@"Age", @"Age"),
                new InputOutputColumnPair(@"Height", @"Height"),
                new InputOutputColumnPair(@"Weight", @"Weight"),
                new InputOutputColumnPair(@"FCVC", @"FCVC"),
                new InputOutputColumnPair(@"NCP", @"NCP"),
                new InputOutputColumnPair(@"CH2O", @"CH2O"),
                new InputOutputColumnPair(@"FAF", @"FAF"),
                new InputOutputColumnPair(@"TUE", @"TUE")
            }, maximumBinCount: 4).Append(mlContext.Transforms.Concatenate(
                @"Features", @"Gender",
                @"family_history_with_overweight", @"FAVC", @"CAEC", @"SMOKE",
                @"SCC", @"CALC", @"MTRANS", @"Age", @"Height", @"Weight",
                @"FCVC", @"NCP", @"CH2O", @"FAF", @"TUE"));
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Concatenate(@"Features", @"Gender",
                @"family_history_with_overweight", @"FAVC", @"CAEC", @"SMOKE",
                @"SCC", @"CALC", @"MTRANS", @"Age", @"Height", @"Weight",
                @"FCVC", @"NCP", @"CH2O", @"FAF", @"TUE");
        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return processingType switch
        {
            ProcessingType.FeatureBinningAndCustomLabelMapping => mlContext
                .Transforms.Conversion.MapValueToKey(@"Label", @"NObeyesdad",
                    keyData: mlContext.Data.LoadFromEnumerable(_lookupData)),
            ProcessingType.Standard => mlContext.Transforms.Conversion
                .MapValueToKey(@"NObeyesdad", @"NObeyesdad",
                    addKeyValueAnnotationsAsText: false)
                .Append(
                    mlContext.Transforms.CopyColumns("Label", "NObeyesdad")),
            _ => throw new NotImplementedException()
        };
    }

    public class ObesityLevelsModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"Gender")]
        public string Gender { get; set; }

        [LoadColumn(1)] [ColumnName(@"Age")] public float Age { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"Height")]
        public float Height { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"Weight")]
        public float Weight { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"family_history_with_overweight")]
        public bool Family_history_with_overweight { get; set; }

        [LoadColumn(5)] [ColumnName(@"FAVC")] public bool FAVC { get; set; }

        [LoadColumn(6)] [ColumnName(@"FCVC")] public float FCVC { get; set; }

        [LoadColumn(7)] [ColumnName(@"NCP")] public float NCP { get; set; }

        [LoadColumn(8)] [ColumnName(@"CAEC")] public string CAEC { get; set; }

        [LoadColumn(9)] [ColumnName(@"SMOKE")] public bool SMOKE { get; set; }

        [LoadColumn(10)] [ColumnName(@"CH2O")] public float CH2O { get; set; }

        [LoadColumn(11)] [ColumnName(@"SCC")] public bool SCC { get; set; }

        [LoadColumn(12)] [ColumnName(@"FAF")] public float FAF { get; set; }

        [LoadColumn(13)] [ColumnName(@"TUE")] public float TUE { get; set; }

        [LoadColumn(14)] [ColumnName(@"CALC")] public string CALC { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"MTRANS")]
        public string MTRANS { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"NObeyesdad")]
        public string NObeyesdad { get; set; }
    }
}