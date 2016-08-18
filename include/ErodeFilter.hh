#ifndef ERODEFILTER_HH_
# define ERODEFILTER_HH_

# include <string>
# include "tbb/pipeline.h"
# include "cv.h"
# include "highgui.h"

class ErodeFilter : public tbb::filter
{
  public:
    ErodeFilter();
    std::string get_name ();
  private:
    void* operator()(void*);
};

#endif /* !ERODEFILTER_HH_ */
