using Italbytz.ML.UCIMLR.Trainers;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace Italbytz.ML.UCIMLR;

public static class ITransformerExtensions
{
    public static ICanSaveModel ExtractModel(this ITransformer transformer)
    {
        IPredictionTransformer<ICanSaveModel>? predictionTransformer = null;
        switch (transformer)
        {
            case IEnumerable<ITransformer> chain:
            {
                foreach (var chainItem in chain)
                    if (chainItem is IPredictionTransformer<ICanSaveModel>
                        predTransformer)
                    {
                        predictionTransformer = predTransformer;
                        break;
                    }

                break;
            }
            case IPredictionTransformer<ICanSaveModel>
                predTransformer:
                predictionTransformer = predTransformer;
                break;
        }

        var model = predictionTransformer.Model;
        if (model is OneVersusAllModelParameters oneVersusAllModelParameters)
            model = oneVersusAllModelParameters.ToPublic();
        return model;
    }
}