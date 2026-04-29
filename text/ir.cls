\newcommand\@IRversion{2018/05/30 v0.1}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ir}
\RequirePackage{ifthen}

\newboolean{@no@authorname}\setboolean{@no@authorname}{false}

\DeclareOption{soumission}{
   \setboolean{@no@authorname}{true}
}

\ProcessOptions\relax
\PassOptionsToClass{10pt,twoside,a4paper}{article}
\LoadClass{article}

\RequirePackage{a4wide}
\RequirePackage[french,linesnumbered,vlined]{algorithm2e}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage[french]{babel}
\RequirePackage[small,hang]{caption2}
\RequirePackage{calc}
\RequirePackage{epsfig}
\RequirePackage[mathcal]{euscript}
\RequirePackage{eurosym}
\RequirePackage{float}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{latexsym}
\RequirePackage[utf8]{inputenc}
\RequirePackage{mathrsfs}
\RequirePackage{multirow}
\RequirePackage{nopageno}
\RequirePackage{pstricks}
\RequirePackage{rotating}
\RequirePackage{subfigure}
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{verbatim}
\RequirePackage{url}
\RequirePackage{xcolor}

\setlength{\textwidth}{13.2cm}
\setlength{\textheight}{19.3cm}
\setlength{\oddsidemargin}{1.36 cm}
\setlength{\evensidemargin}{1.36 cm}
\setlength{\topmargin}{1.48 cm}
\clubpenalty=6000
\widowpenalty=6000

\RequirePackage{fancyhdr}
\pagestyle{fancy}

\newcommand{\doublerule}[1][.4pt]{%
  \noindent
  \makebox[0pt][l]{\rule[.7ex]{\linewidth}{#1}}%
  \rule[.3ex]{\linewidth}{#1}}

\fancyhead{} \fancyfoot{}
\newcommand\@titre{Titre court \`{a} d\'{e}finir avec \texttt{$\backslash$titre\{Votre titre\}} dans le pr\'{e}ambule}
\newcommand\@auteur{}
\newcommand\@affiliation{}
\fancyhead[LE]{Titre court \`{a} d\'{e}finir avec \texttt{$\backslash$titrecourt\{...\}}}
\newcommand*\titrecourt[1]{\fancyhead[LE]{#1}}
\fancyhead[RO]{Liste courte des auteurs \`{a} d\'{e}finir avec \texttt{$\backslash$nomcourt\{...\}}}
\newcommand*\nomcourt[1]{\fancyhead[RO]{%
        \ifthenelse{\boolean{@no@authorname}}{\newbox\b@nomcourt\setbox\b@nomcourt=\hbox{#1}\hskip\wd\b@nomcourt}{#1}}}
\newcommand\@the@page@{~}
\renewcommand\@the@page@{\thepage}
\fancyfoot[LE,RO]{Projet IR \the\year{} -- \@the@page@}
\newcommand\titre[1]{\renewcommand\@titre{#1}}
\newcommand\auteur[1]{\renewcommand\@auteur{#1}}%
\newcommand\affiliation[1]{\renewcommand\@affiliation{#1}}
\newcommand*\affilsep{$^,$}
\newcommand*\affil[1]{%
   $^{\ifcase#1\or{*}\or{**}\or{***}\or{****}
               \or{\sharp}\or{\sharp\sharp}\or{\sharp\sharp\sharp}
               \or{\sharp\sharp\sharp\sharp}\fi}$}
\newcommand\http[1]{\noextrasfrench{#1}\extrasfrench}

\newcommand\gen@titre{
    \vspace*{4mm}\noindent%
    \parbox{\linewidth}{\center\Large%
       \textbf{\@titre}}%
    \\[1ex]
    \begin{center}
    \ifthenelse{\boolean{@no@authorname}}{%
       \newbox\b@entete\setbox\b@entete=\vbox{%
        \@auteur%
        \\[3ex]
        \begin{tabular}{c}
        \@affiliation
        \end{tabular}
        }\vskip\ht\b@entete \vskip\dp\b@entete%}
		}
        {%
        \@auteur%
        \\[3ex]
        \begin{tabular}{c}
        \@affiliation
        \end{tabular}%
        \\[3ex]%
	\@@encadrant%
        }
    \vspace{1ex}
    \end{center}
}

\newcommand\@motcle{Donner le texte qui doit figurer ici
    avec la commande
    \texttt{$\backslash$motcle\{Liste des mots clés, séparés par des virgules\}}
    dans le pr\'{e}ambule
    (avant le \texttt{$\backslash$begin\{document\}}).
    }
\newcommand\motcle[1]{\renewcommand\@motcle{#1}}
\newcommand\@@motcle{{\small\textbf{Mots-clés~: }\@motcle}\\}

\newcommand\@resume{Donner le texte qui doit figurer ici
    avec la commande
    \texttt{$\backslash$resume\{Votre r\'{e}sume\ldots{}\}}
    dans le pr\'{e}ambule
    (avant le \texttt{$\backslash$begin\{document\}}).
    }
\newcommand\resume[1]{\renewcommand\@resume{#1}}
\newcommand\@@resume{\begin{abstract}\@resume\\[3ex]\@@motcle\end{abstract}%
  }

\newcommand\@encadrant{Donner le texte qui doit figurer ici
    avec la commande
    \texttt{$\backslash$encadrant\{Prénoms et noms des encadrants\}}
    dans le pr\'{e}ambule
    (avant le \texttt{$\backslash$begin\{document\}}).
    }
\newcommand\encadrant[1]{\renewcommand\@encadrant{#1}}
\newcommand\@@encadrant{{\footnotesize\textbf{Travail encadré par~: }\@encadrant}\\}

\newcommand\remerciement[1]{~\\[3ex]\noindent\small\textbf{Remerciements~: }#1}

\AtBeginDocument{%
  \thispagestyle{empty}%
  \doublerule%
  \gen@titre%
  \@@resume%
  \doublerule%
}

\renewcommand{\captionfont}{\it \small}
\renewcommand{\captionlabelfont}{\it \bf \small}
\renewcommand{\captionlabeldelim}{\ :}

\floatstyle{ruled}%
\newfloat{algorithme}{!htt}{lob}%
\floatname{algorithme}{Algorithme}%
\SetKw{KwA}{{\bf à}}
\SetKw{KwNon}{{\bf non}}
\SetKw{KwOu}{{\bf ou}}