using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    CDCDiabetesDataset : Dataset<CDCDiabetesDataset.CDCDiabetesModelInput>
{
    private readonly LookupMap<float>[] _lookupData =
    [
        new(0.0f),
        new(1.0f)
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.CDC_Diabetes_Health_Indicators.csv";

    public override string FilePrefix { get; } = "cdcd";

    public override string? LabelColumnName { get; } =
        @"Diabetes_binary";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "HighBP",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "HighChol",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "CholCheck",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "BMI",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Smoker",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Stroke",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "HeartDiseaseorAttack",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "PhysActivity",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Fruits",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Veggies",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "HvyAlcoholConsump",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "AnyHealthcare",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "NoDocbcCost",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "GenHlth",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "MentHlth",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "PhysHlth",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "DiffWalk",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Sex",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
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
            "ColumnName": "Education",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Income",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Diabetes_binary",
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
        return LoadFromTextFile<CDCDiabetesModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return mlContext.Transforms.ReplaceMissingValues(new[]
        {
            new InputOutputColumnPair(@"HighBP", @"HighBP"),
            new InputOutputColumnPair(@"HighChol", @"HighChol"),
            new InputOutputColumnPair(@"CholCheck", @"CholCheck"),
            new InputOutputColumnPair(@"BMI", @"BMI"),
            new InputOutputColumnPair(@"Smoker", @"Smoker"),
            new InputOutputColumnPair(@"Stroke", @"Stroke"),
            new InputOutputColumnPair(@"HeartDiseaseorAttack",
                @"HeartDiseaseorAttack"),
            new InputOutputColumnPair(@"PhysActivity", @"PhysActivity"),
            new InputOutputColumnPair(@"Fruits", @"Fruits"),
            new InputOutputColumnPair(@"Veggies", @"Veggies"),
            new InputOutputColumnPair(@"HvyAlcoholConsump",
                @"HvyAlcoholConsump"),
            new InputOutputColumnPair(@"AnyHealthcare", @"AnyHealthcare"),
            new InputOutputColumnPair(@"NoDocbcCost", @"NoDocbcCost"),
            new InputOutputColumnPair(@"GenHlth", @"GenHlth"),
            new InputOutputColumnPair(@"MentHlth", @"MentHlth"),
            new InputOutputColumnPair(@"PhysHlth", @"PhysHlth"),
            new InputOutputColumnPair(@"DiffWalk", @"DiffWalk"),
            new InputOutputColumnPair(@"Sex", @"Sex"),
            new InputOutputColumnPair(@"Age", @"Age"),
            new InputOutputColumnPair(@"Education", @"Education"),
            new InputOutputColumnPair(@"Income", @"Income")
        });
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
                new InputOutputColumnPair(@"BMI", @"BMI"),
                new InputOutputColumnPair(@"MentHlth", @"MentHlth"),
                new InputOutputColumnPair(@"PhysHlth", @"PhysHlth"),
                new InputOutputColumnPair(@"Age", @"Age")
            }, maximumBinCount: 4).Append(mlContext.Transforms.Concatenate(
                @"Features", @"HighBP",
                @"HighChol", @"CholCheck", @"BMI", @"Smoker", @"Stroke",
                @"HeartDiseaseorAttack", @"PhysActivity", @"Fruits", @"Veggies",
                @"HvyAlcoholConsump", @"AnyHealthcare", @"NoDocbcCost",
                @"GenHlth", @"MentHlth", @"PhysHlth", @"DiffWalk", @"Sex",
                @"Age", @"Education", @"Income"));
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Concatenate(@"Features", @"HighBP",
                @"HighChol", @"CholCheck", @"BMI", @"Smoker", @"Stroke",
                @"HeartDiseaseorAttack", @"PhysActivity", @"Fruits", @"Veggies",
                @"HvyAlcoholConsump", @"AnyHealthcare", @"NoDocbcCost",
                @"GenHlth", @"MentHlth", @"PhysHlth", @"DiffWalk", @"Sex",
                @"Age", @"Education", @"Income");
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
                .Transforms.Conversion.MapValueToKey(@"Label",
                    @"Diabetes_binary",
                    keyData: mlContext.Data.LoadFromEnumerable(_lookupData)),
            ProcessingType.Standard => mlContext.Transforms.Conversion
                .MapValueToKey(@"Diabetes_binary", @"Diabetes_binary",
                    addKeyValueAnnotationsAsText: false)
                .Append(
                    mlContext.Transforms.CopyColumns("Label",
                        "Diabetes_binary")),
            _ => throw new NotImplementedException()
        };
    }

    public class CDCDiabetesModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"HighBP")]
        public float HighBP { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"HighChol")]
        public float HighChol { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"CholCheck")]
        public float CholCheck { get; set; }

        [LoadColumn(3)] [ColumnName(@"BMI")] public float BMI { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"Smoker")]
        public float Smoker { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"Stroke")]
        public float Stroke { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"HeartDiseaseorAttack")]
        public float HeartDiseaseorAttack { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"PhysActivity")]
        public float PhysActivity { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"Fruits")]
        public float Fruits { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"Veggies")]
        public float Veggies { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"HvyAlcoholConsump")]
        public float HvyAlcoholConsump { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"AnyHealthcare")]
        public float AnyHealthcare { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"NoDocbcCost")]
        public float NoDocbcCost { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"GenHlth")]
        public float GenHlth { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"MentHlth")]
        public float MentHlth { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"PhysHlth")]
        public float PhysHlth { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"DiffWalk")]
        public float DiffWalk { get; set; }

        [LoadColumn(17)] [ColumnName(@"Sex")] public float Sex { get; set; }

        [LoadColumn(18)] [ColumnName(@"Age")] public float Age { get; set; }

        [LoadColumn(19)]
        [ColumnName(@"Education")]
        public float Education { get; set; }

        [LoadColumn(20)]
        [ColumnName(@"Income")]
        public float Income { get; set; }

        [LoadColumn(21)]
        [ColumnName(@"Diabetes_binary")]
        public float Diabetes_binary { get; set; }
    }
}