# Reporting Helper Functions

required_packages = c('dplyr', 'knitr', 'ggplot2', 'xtable', 'stringr', 'reshape2', 'scales', 'gridExtra', 'grid')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}


open.pdf  = function(pdf_file){
    system(sprintf("open \"%s\"", pdf_file))
}

merge.pdfs = function(files_to_merge, merged_file_name = "report.pdf", open_after = TRUE){
    file_exists = array(sapply(files_to_merge, file.exists))
    files_to_merge = files_to_merge[file_exists]
    system(sprintf("pdftk \"%s\" cat output \"%s\"",paste(files_to_merge,collapse='\" \"'),merged_file_name))
    if (open_after){
        open.pdf(merged_file_name)
    }
    return(merged_file_name)
}

jam.pdfs =  function(files_to_merge, merged_file_name="report.pdf", rows = NULL, cols = 2, landscape = FALSE, open_after=FALSE, delete_originals=FALSE){

    file_exists = array(sapply(files_to_merge, file.exists))
    stopifnot(all(file_exists))

    n_files = length(files_to_merge);
    if (n_files > 0){

        if (is.null(rows)){
            rows = n_files %/% cols;
        }
        if (!grepl("*.pdf", merged_file_name)){
            merged_file_name = paste0(merged_file_name,".pdf");
        }
        size_string = sprintf("%dx%d",cols, rows)
        files_string = paste(files_to_merge, collapse='\" \"');

        if (landscape){
            system(sprintf("pdfjam \"%s\" --nup %s --landscape --outfile \"%s\"",files_string, size_string, merged_file_name))
        } else {
            system(sprintf("pdfjam \"%s\" --nup %s --no-landscape --outfile \"%s\"",files_string, size_string, merged_file_name))
        }

        if (delete_originals){
            file.remove(files_to_merge);
        }

        if (open_after){
            system(sprintf("open \"%s\"", merged_file_name))
        }
        return(merged_file_name)
    }
}



human.numbers = function(x = NULL, smbl =""){
    #https://github.com/fdryan/R/blob/master/ggplot2_formatter.r
    humanity <- function(y){

        if (!is.na(y)){

            b <- round(abs(y) / 1e9, 0.1)
            m <- round(abs(y) / 1e6, 0.1)
            k <- round(abs(y) / 1e3, 0.1)

            if ( y >= 0 ){
                y_is_positive <- ""
            } else {
                y_is_positive <- "-"
            }

            if ( k < 1 ) {
                paste0(y_is_positive, smbl, y )
            } else if ( m < 1){
                paste0 (y_is_positive, smbl,  k , "K")
            } else if (b < 1){
                paste0 (y_is_positive, smbl, m ,"M")
            } else {
                paste0 (y_is_positive, smbl,  comma(b), "N")
            }
        }
    }
    sapply(x,humanity)
}

default.plot.theme = function(){

    line_color = "#E9E9E9";

    default_theme = theme_bw() +
        theme(title = element_text(size = 18),
              plot.margin = margin(t = 0.25, r = 0, b = 0.75, l = 0.25, unit = "cm"),
              axis.line = element_blank(),
              panel.border = element_rect(size = 2.0, color = line_color),
              panel.grid.minor = element_blank(),
              panel.grid.major = element_line(linetype="solid", size=1.0, color=line_color),
              #
              axis.title.x = element_text(size = 20, margin = margin(t = 20, unit = "pt")),
              axis.text.x   = element_text(size = 20),
              axis.ticks.x  = element_line(size = 1.0, color = line_color),
              #
              axis.title.y = element_text(size = 20, margin = margin(b = 20, unit = "pt")),
              axis.text.y   = element_text(size=20),
              axis.ticks.y	= element_line(size=1.0, color = line_color),
              #
              legend.position="none",
              legend.title = element_blank(),
              legend.text = element_text(face="plain",size=14,angle=0,lineheight=30),
              #legend.key.width = unit(1.5, "cm"),
              #legend.key.height = unit(1.5, "cm"),
              #legend.text.align = 0,
              legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"))

    return(default_theme);
}
