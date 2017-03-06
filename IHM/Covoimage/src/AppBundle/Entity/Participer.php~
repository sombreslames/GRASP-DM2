<?php
namespace AppBundle\Entity;
use Symfony\Component\Validator\Constraints as CSTRT;
use Doctrine\ORM\Mapping as ORM;

/**
 * @ORM\Entity
 * @ORM\Table(name="Participer")
 * relation n,n entre la table Personne et Trajet
 */
 class Participer
{
	/**
	 * @ORM\Column(name="nbplaces", type="integer")
	 */
	private $nbplaces;
	/**
	 * @ORM\Id
	 * @ORM\Column(name="ID_trajet", type="integer")
	 * @ORM\ManyToOne(targetEntity="trajet")
     * @ORM\JoinColumn(name="ID_trajet", referencedColumnName="ID_trajet")
	 */
	private $ID_trajet;
	/**
	 * @ORM\Id
	 * @ORM\Column(name="ID_personne", type="integer")
	 * @ORM\ManyToOne(targetEntity="Personne")
     * @ORM\JoinColumn(name="ID_personne", referencedColumnName="ID_personne")
	 */
	private $ID_personne;

    /**
     * Set nbplaces
     *
     * @param integer $nbplaces
     *
     * @return Participer
     */
    public function setNbplaces($nbplaces)
    {
        $this->nbplaces = $nbplaces;

        return $this;
    }

    /**
     * Get nbplaces
     *
     * @return integer
     */
    public function getNbplaces()
    {
        return $this->nbplaces;
    }

    /**
     * Set iDTrajet
     *
     * @param integer $iDTrajet
     *
     * @return Participer
     */
    public function setIDTrajet($iDTrajet)
    {
        $this->ID_trajet = $iDTrajet;

        return $this;
    }

    /**
     * Get iDTrajet
     *
     * @return integer
     */
    public function getIDTrajet()
    {
        return $this->ID_trajet;
    }

    /**
     * Set iDPersonne
     *
     * @param integer $idpersonne
     *
     * @return Participer
     */
    public function setIDPersonne($iDPersonne)
    {
        $this->ID_personne = $iDPersonne;

        return $this;
    }

    /**
     * Get iDPersonne
     *
     * @return integer
     */
    public function getIDPersonne()
    {
        return $this->ID_personne;
    }

    /**
     * Set description
     *
     * @param string $description
     *
     * @return Participer
     */
    public function setDescription($description)
    {
        $this->description = $description;

        return $this;
    }

    /**
     * Get description
     *
     * @return string
     */
    public function getDescription()
    {
        return $this->description;
    }
 }
