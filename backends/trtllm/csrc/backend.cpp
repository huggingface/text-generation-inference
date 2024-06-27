#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

# ifdef __cplusplus
extern "C" {
#endif

#if defined __GNUC__
#define TGI_BACKEND_PUBLIC __attribute__ ((visibility("default")))
#else
#pragma message ("Compiler does not support symbol visibility.")
#define TGI_BACKEND_PUBLIC
#endif

struct tgi_trtllm_backend;

/***
 * Create and allocate all the required resources to run inference over the provided engine
 * @return A `tgi_trtllm_backend` handle
 */
TGI_BACKEND_PUBLIC struct tgi_trtllm_backend *tgi_trtllm_backend_create();

/***
 * Free-up all the resources associated with this backend
 * @param handle `tgi_trtllm_backend` handle
 */
TGI_BACKEND_PUBLIC void tgi_trtllm_backend_destroy(struct tgi_trtllm_backend *handle);

/***
 *
 * @param handle `tgi_trtllm_backend` handle
 */
TGI_BACKEND_PUBLIC void tgi_trtllm_backend_submit(struct tgi_trtllm_backend *handle);

#ifdef __cplusplus
};
#endif