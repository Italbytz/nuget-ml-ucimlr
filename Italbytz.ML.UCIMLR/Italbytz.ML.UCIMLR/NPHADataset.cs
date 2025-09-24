using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class NPHADataset : Dataset<NPHADataset.NationalPollModelInput>
{
    private readonly LookupMap<float>[] _lookupData =
    [
        new(1.0f),
        new(2.0f),
        new(3.0f)
    ];

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.national_poll_on_healthy_aging_npha.csv";

    public override string FilePrefix { get; } = "npha";

    public override string? LabelColumnName { get; } =
        @"Number_of_Doctors_Visited";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "Age",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Physical_Health",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Mental_Health",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Dental_Health",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Employment",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Stress_Keeps_Patient_from_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Medication_Keeps_Patient_from_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Pain_Keeps_Patient_from_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Bathroom_Needs_Keeps_Patient_from_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Uknown_Keeps_Patient_from_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Trouble_Sleeping",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Prescription_Sleep_Medication",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Race",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Gender",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Number_of_Doctors_Visited",
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
        return LoadFromTextFile<NationalPollModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
            return mlContext.Transforms.Concatenate(@"Features", @"Age",
                @"Physical_Health", @"Mental_Health", @"Dental_Health",
                @"Employment", @"Stress_Keeps_Patient_from_Sleeping",
                @"Medication_Keeps_Patient_from_Sleeping",
                @"Pain_Keeps_Patient_from_Sleeping",
                @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                @"Uknown_Keeps_Patient_from_Sleeping", @"Trouble_Sleeping",
                @"Prescription_Sleep_Medication", @"Race", @"Gender");
        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
        {
            if (processingType ==
                ProcessingType.FeatureBinningAndCustomLabelMapping)
                return mlContext.Transforms.Conversion.MapValueToKey(
                    @"Label", @"Number_of_Doctors_Visited",
                    keyData: mlContext.Data.LoadFromEnumerable(_lookupData));


            if (processingType == ProcessingType.Standard)
                return mlContext.Transforms.Conversion.MapValueToKey(
                    @"Number_of_Doctors_Visited",
                    @"Number_of_Doctors_Visited",
                    addKeyValueAnnotationsAsText: false).Append(
                    mlContext.Transforms.CopyColumns("Label",
                        "Number_of_Doctors_Visited"));
        }

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
                new InputOutputColumnPair(@"Age", @"Age"),
                new InputOutputColumnPair(@"Physical_Health",
                    @"Physical_Health"),
                new InputOutputColumnPair(@"Mental_Health",
                    @"Mental_Health"),
                new InputOutputColumnPair(@"Dental_Health",
                    @"Dental_Health"),
                new InputOutputColumnPair(@"Employment", @"Employment"),
                new InputOutputColumnPair(
                    @"Stress_Keeps_Patient_from_Sleeping",
                    @"Stress_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Medication_Keeps_Patient_from_Sleeping",
                    @"Medication_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Pain_Keeps_Patient_from_Sleeping",
                    @"Pain_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Uknown_Keeps_Patient_from_Sleeping",
                    @"Uknown_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Trouble_Sleeping",
                    @"Trouble_Sleeping"),
                new InputOutputColumnPair(@"Prescription_Sleep_Medication",
                    @"Prescription_Sleep_Medication"),
                new InputOutputColumnPair(@"Race", @"Race"),
                new InputOutputColumnPair(@"Gender", @"Gender")
            });

        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer>? BuildLabelRemappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
        {
            if (processingType ==
                ProcessingType.FeatureBinningAndCustomLabelMapping) return null;
            return mlContext.Transforms.Conversion.MapKeyToValue(
                @"PredictedLabel", @"PredictedLabel");
        }

        throw new NotImplementedException();
    }

    public class NationalPollModelInput
    {
        [LoadColumn(0)] [ColumnName(@"Age")] public float Age { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"Physical_Health")]
        public float Physical_Health { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"Mental_Health")]
        public float Mental_Health { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"Dental_Health")]
        public float Dental_Health { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"Employment")]
        public float Employment { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"Stress_Keeps_Patient_from_Sleeping")]
        public float Stress_Keeps_Patient_from_Sleeping { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"Medication_Keeps_Patient_from_Sleeping")]
        public float Medication_Keeps_Patient_from_Sleeping { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"Pain_Keeps_Patient_from_Sleeping")]
        public float Pain_Keeps_Patient_from_Sleeping { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"Bathroom_Needs_Keeps_Patient_from_Sleeping")]
        public float Bathroom_Needs_Keeps_Patient_from_Sleeping { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"Uknown_Keeps_Patient_from_Sleeping")]
        public float Uknown_Keeps_Patient_from_Sleeping { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"Trouble_Sleeping")]
        public float Trouble_Sleeping { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"Prescription_Sleep_Medication")]
        public float Prescription_Sleep_Medication { get; set; }

        [LoadColumn(12)] [ColumnName(@"Race")] public float Race { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"Gender")]
        public float Gender { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"Number_of_Doctors_Visited")]
        public float Number_of_Doctors_Visited { get; set; }
    }
}