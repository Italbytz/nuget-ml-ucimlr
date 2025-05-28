using System.Collections.Immutable;
using Microsoft.ML;

namespace Italbytz.ML.UCIMLR.Trainers;

public class
    PublicOneVersusAllModelParameters : ICanSaveModel
{
    public ImmutableArray<object> SubModelParameters { get; set; }

    public void Save(ModelSaveContext ctx)
    {
        throw new NotImplementedException();
    }
}