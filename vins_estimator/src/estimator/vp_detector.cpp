#include "vp_detector.h"

std::vector<double> vp_detector(cv::Mat img)
{
    cv::Mat detected_edges;

    // equalize histogram
    cv::equalizeHist(img, img);

    // Detect lines using the LineSegmentDetector
    std::vector<cv::Vec4i> lines_std;
    //cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    Ptr<cv::ximgproc::FastLineDetector> ls = cv::ximgproc::createFastLineDetector();	
    ls->detect(img, lines_std);

    // Filter lines based on length and vertical constraints
    std::vector<std::vector<int>> points;
    std::vector<cv::Vec4i> final_lines; 
    int minlength = img.cols * img.cols * 0.001;
    for (size_t i = 0; i < lines_std.size(); i++)
    {
        // Ignore almost vertical or horizontal lines
        if (abs(lines_std[i][0] - lines_std[i][2]) < 10 || abs(lines_std[i][1] - lines_std[i][3]) < 10)
            continue;

        // Ignore shorter lines
        if (((lines_std[i][0] - lines_std[i][2]) * (lines_std[i][0] - lines_std[i][2]) + 
             (lines_std[i][1] - lines_std[i][3]) * (lines_std[i][1] - lines_std[i][3])) < minlength)
            continue;

        // Store valid lines for further processing
        std::vector<int> temp = {lines_std[i][0], lines_std[i][1], lines_std[i][2], lines_std[i][3]};
        points.push_back(temp);
        final_lines.push_back(lines_std[i]);
    }
    
    // Draw the filtered lines on the image
    for (size_t i = 0; i < final_lines.size(); i++)
    {
        cv::line(img, 
                 cv::Point(final_lines[i][0], final_lines[i][1]), 
                 cv::Point(final_lines[i][2], final_lines[i][3]), 
                 cv::Scalar(255, 0, 0), // Color of the line (blue in this example)
                 2, // Line thickness
                 cv::LINE_AA); // Anti-aliased line for better quality
    }

    // Initialize Armadillo matrices A and b for line equation Ax = b
    arma::mat A(points.size(), 2, arma::fill::zeros);
    arma::mat b(points.size(), 1, arma::fill::zeros);
    for (size_t i = 0; i < points.size(); i++)
    {
        A(i, 0) = -(points[i][3] - points[i][1]); // -(y2 - y1)
        A(i, 1) = (points[i][2] - points[i][0]);  // (x2 - x1)
        b(i, 0) = A(i, 0) * points[i][0] + A(i, 1) * points[i][1]; // -(y2 - y1) * x1 + (x2 - x1) * y1
    }

    // Estimate vanishing point using error minimization
    arma::mat soln(2, 1, arma::fill::zeros);
    double min_err = std::numeric_limits<double>::max();
    arma::mat res;
    
    for (size_t i = 0; i < points.size(); i++)
    {
        for (size_t j = i + 1; j < points.size(); j++)
        {
            arma::uvec indices = {i, j};
            arma::mat Atemp = A.rows(indices);
            arma::mat btemp = b.rows(indices);

            // Check if lines are parallel (rank of A should be 2)
            if (arma::rank(Atemp) != 2)
                continue;

            // Solve the system A * x = b
            res = arma::solve(Atemp, btemp);
            arma::mat error = A * res - b;
            double total_error = arma::accu(arma::abs(error));

            // Find the solution with minimum error
            if (total_error < min_err)
            {
                min_err = total_error;
                soln = res;
            }
        }
    }

    std::vector<double> van_point = {soln(0),soln(1)};
    return van_point;
}