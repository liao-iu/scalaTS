logLevel := Level.Warn

resolvers += Resolver.url("plugins-artifactory", url("http://scalasbt." +
    "artifactoryonline.com/scalasbt/sbt-plugin-releases"))(Resolver.ivyStylePatterns)

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.1")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "0.8.0")