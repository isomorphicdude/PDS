import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import sys



config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_bool("compute_fid", False, "Whether to compute FID.")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results.")
flags.DEFINE_float("speed_up",1,"The times of speedup.")
flags.DEFINE_float("alpha",5,"The parameter alpha.")
flags.DEFINE_string("freq_mask_path",None,"The path of the frequency mask.")
flags.DEFINE_string("space_mask_path",None,"The path of the spatial mask.")
flags.DEFINE_float("sde_solver_lr",1.2720,"The learning rate of the sde solver.")
flags.DEFINE_bool("verbose",False,"Whether to print the details of the evaluation.")
flags.mark_flags_as_required(["workdir", "config"])

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main(args):
    if not FLAGS.compute_fid:
        run_lib.evaluate(FLAGS.config,
                     FLAGS.workdir,
                     FLAGS.eval_folder,
                     FLAGS.speed_up,
                     FLAGS.freq_mask_path,
                     FLAGS.space_mask_path,
                     FLAGS.alpha,
                     FLAGS.sde_solver_lr,
                     FLAGS.verbose)
    else:
        run_lib.evaluate_fid(FLAGS.config,
                        FLAGS.workdir,
                        FLAGS.eval_folder,
                        FLAGS.speed_up,
                        FLAGS.freq_mask_path,
                        FLAGS.space_mask_path,
                        FLAGS.alpha,
                        FLAGS.sde_solver_lr,
                        FLAGS.verbose)

if __name__ == "__main__":
    app.run(main)